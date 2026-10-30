"""
Serves the dashboard and its API.

  GET  /                        dashboard
  GET  /dashboard.js            dashboard script (served same-origin so CSP can forbid inline JS)
  GET  /healthz                 liveness + mode + event-write failure count
  GET  /state[?replay=<file>]   live (or replayed, read-only) state
  GET  /replays                 list saved replays
  GET  /leaderboard[?sort=...]  aggregated record across every completed run
  GET  /export/events.json|csv[?replay=<file>]   raw events
  GET  /export/summary.json[?replay=<file>]      derived metrics + config + seed
  POST /inject_task             "ask the economy" (rate limited)
  POST /control/pause|resume|speed|save_replay|reset      (privileged, see below)

ACCESS MODES (core/config.py)
  local (default)   controls open, binds 127.0.0.1
  CONTROL_TOKEN=x   controls require header  X-Control-Token: x
  PUBLIC_MODE=1     fail closed: controls disabled (403) unless CONTROL_TOKEN is set

SINGLE-PROCESS BY DESIGN: the simulation thread, control state and event log
live in this process. Run ONE worker process (threads are fine). See
docs/DEPLOYMENT.md.
"""
import csv
import hmac
import io
import json
import logging
import re
import signal
import threading
import time
from collections import defaultdict, deque

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from core import config, events
from core.control import control
from core.injected_tasks import injected_tasks
from core.runner import manager
from core.security import redact
from core.simconfig import ConfigError
from core.report import build_report
from core.state import build_state
from core.stats import safe_div

log = logging.getLogger("proof_of_work.server")

from pathlib import Path
EXPERIMENTS_DIR = Path(__file__).resolve().parent / "data" / "experiments"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,59}$")

app = Flask(__name__, static_folder=None)
WEB_DIR = "web"


# ------------------------------------------------------------------ plumbing

def _origin_allowed(origin: str) -> bool:
    from urllib.parse import urlparse
    origin = origin.rstrip("/")
    if origin in config.ALLOWED_ORIGINS:
        return True
    try:
        return urlparse(origin).netloc == request.host and urlparse(origin).scheme in ("http", "https")
    except ValueError:
        return False


@app.before_request
def _request_guards():
    """Runs before every request.
      * body size: enforced by Flask (MAX_CONTENT_LENGTH) *and* checked up front;
        waitress also rejects oversized bodies before they reach the app.
      * CSRF: a browser always sends Origin on a cross-site POST. A POST whose
        Origin is not this server's own host (or ALLOWED_ORIGINS) is refused, so
        a malicious web page cannot drive /control/* of a local open-mode server
        with a "simple" cross-origin request. Non-browser clients send no Origin.
      * read endpoints are rate limited per client.
    """
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_REQUEST_BYTES
    length = request.content_length
    if length is not None and length > config.MAX_REQUEST_BYTES:
        return jsonify({"ok": False, "error": "request too large"}), 413
    if request.method == "POST":
        origin = request.headers.get("Origin")
        if origin is not None and not _origin_allowed(origin):
            return jsonify({"ok": False, "error": "cross-origin request refused"}), 403
        if request.headers.get("Sec-Fetch-Site") == "cross-site" and origin is None:
            return jsonify({"ok": False, "error": "cross-site request refused"}), 403
    elif request.path in ("/state", "/leaderboard", "/replays", "/events", "/experiments") or \
            request.path.startswith("/export/"):
        ok, wait = limiter.allow(f"read:{request.remote_addr}", config.READ_MAX_PER_MINUTE, 60)
        if not ok:
            resp = jsonify({"ok": False, "error": "rate limit exceeded"})
            resp.status_code = 429
            resp.headers["Retry-After"] = str(int(wait) + 1)
            return resp


@app.errorhandler(413)
def _too_large(_e):
    return jsonify({"ok": False, "error": "request too large"}), 413


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    resp.headers.setdefault("Content-Security-Policy", (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline' "
        "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'; object-src 'none'"))
    if request.path != "/" and request.path != "/dashboard.js":
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(HTTPException)
def _http_error(e):
    return jsonify({"ok": False, "error": e.name.lower()}), e.code


@app.errorhandler(Exception)
def _unhandled(e):
    # full detail goes to the server log only; the client gets a generic message
    log.error("unhandled error on %s %s: %s", request.method, request.path, redact(e), exc_info=True)
    return jsonify({"ok": False, "error": "internal server error"}), 500


def _json_body() -> dict:
    if not request.data:
        return {}
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise _BadRequest("body must be a JSON object")
    return body


class _BadRequest(Exception):
    pass


@app.errorhandler(_BadRequest)
def _bad_request(e):
    return jsonify({"ok": False, "error": str(e)}), 400


def controls_mode() -> str:
    """'open' | 'token' | 'disabled'"""
    if config.CONTROL_TOKEN:
        return "token"
    return "disabled" if config.PUBLIC_MODE else "open"


def _requested_run_id() -> str | None:
    """run_id the client says it is acting on: X-Run-Id header, else a string
    `run_id` in a JSON-object body. Anything else is ignored."""
    hdr = request.headers.get("X-Run-Id")
    if hdr:
        return hdr
    body = request.get_json(silent=True) if request.data else None
    value = body.get("run_id") if isinstance(body, dict) else None
    return value if isinstance(value, str) else None


def require_control(fn):
    """Privileged endpoints. Order matters: (1) mode check, (2) throttle repeated
    wrong tokens, (3) constant-time token check, (4) stale-run guard.

    Stale-run guard: a client may send the run_id it is looking at. If a different
    run is now current the request is refused with 409, so a late/stale click or a
    replayed request can never act on a run the user did not see. /control/reset is
    exempt (it starts a new run).
    """
    from functools import wraps

    @wraps(fn)
    def wrapper(*a, **kw):
        mode = controls_mode()
        if mode == "disabled":
            return jsonify({"ok": False, "error": "controls are disabled in public mode "
                            "(set CONTROL_TOKEN to enable them)"}), 403
        if mode == "token":
            blocked, wait = limiter.peek(f"authfail:{request.remote_addr}",
                                         config.AUTH_FAIL_MAX_PER_MINUTE, 60)
            if blocked:
                resp = jsonify({"ok": False, "error": "too many failed attempts, try again later"})
                resp.status_code = 429
                resp.headers["Retry-After"] = str(int(wait) + 1)
                return resp
            supplied = request.headers.get("X-Control-Token", "")
            if not hmac.compare_digest(supplied.encode(), config.CONTROL_TOKEN.encode()):
                limiter.allow(f"authfail:{request.remote_addr}", config.AUTH_FAIL_MAX_PER_MINUTE, 60)
                return jsonify({"ok": False, "error": "missing or invalid control token"}), 401
        if fn.__name__ != "control_reset":
            wanted = _requested_run_id()
            if wanted and wanted != events.current_run_id():
                return jsonify({"ok": False, "error": "stale request: a different run is now active",
                                "current_run_id": events.current_run_id()}), 409
        return fn(*a, **kw)
    return wrapper


class SlidingWindowLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._hits = defaultdict(deque)

    def allow(self, key: str, limit: int, window: float) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False, window - (now - q[0])
            q.append(now)
            if len(self._hits) > 5000:  # bound memory
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True, 0.0

    def peek(self, key: str, limit: int, window: float) -> tuple[bool, float]:
        """(is_blocked, seconds_until_unblocked) WITHOUT recording a hit."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            return (len(q) >= limit), (window - (now - q[0]) if q else 0.0)

    def reset(self):
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def _load_events(replay_arg):
    """(events, error_response|None). Replays are read-only views."""
    if replay_arg:
        try:
            path = events.replay_path(replay_arg)
        except ValueError:
            return None, (jsonify({"ok": False, "error": "invalid replay name"}), 400)
        if not path.exists():
            return None, (jsonify({"ok": False, "error": "replay not found"}), 404)
        return events.read_replay(replay_arg), None
    return events.read_all(), None


# ------------------------------------------------------------------- static

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "dashboard.html")


@app.route("/dashboard.js")
def dashboard_js():
    return send_from_directory(WEB_DIR, "dashboard.js", mimetype="text/javascript")


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "mode": config.mode(), "simulation_running": manager.is_running(),
                    "event_write_failures": events.write_failures,
                    "config_warnings": config.config_warnings()})


# ------------------------------------------------------------------- read API

@app.route("/state")
def state():
    all_events, err = _load_events(request.args.get("replay"))
    if err:
        return err
    st = build_state(all_events)
    replay = bool(request.args.get("replay"))
    st["replay"] = replay
    st["paused"] = False if replay else control.is_paused()
    st["round_delay"] = control.round_delay
    st["controls"] = controls_mode()
    st["pending_questions"] = injected_tasks.pending_count()
    if replay:
        st["status"] = "replay"
        st["mode"] = st["mode"] or "unknown"
    elif st["mode"] is None:
        st["mode"] = config.mode()
    return jsonify(st)


@app.route("/replays")
def replays():
    return jsonify(events.list_replays())


_SORTS = {"win_rate": "win_rate", "balance": "avg_final_balance", "accuracy": "avg_accuracy",
          "profit": "avg_net_pl", "efficiency": "efficiency", "cost": "cost_per_correct"}
_ASCENDING = {"cost_per_correct"}          # lower is better


@app.route("/leaderboard")
def leaderboard():
    sort_key = _SORTS.get(request.args.get("sort", "win_rate"))
    if sort_key is None:
        return jsonify({"ok": False, "error": f"sort must be one of {sorted(_SORTS)}"}), 400
    def _num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")

    # keep only rows whose winner has a valid numeric balance, so wins can never
    # exceed runs (win_rate <= 100%) even if the file contains garbage
    entries = []
    for e in events.read_leaderboard():
        fb = e.get("final_balances")
        if isinstance(fb, dict) and isinstance(e.get("winner"), str) and _num(fb.get(e["winner"])):
            entries.append(e)
    wins, runs, total_bal = defaultdict(int), defaultdict(int), defaultdict(int)
    acc_c, acc_g = defaultdict(int), defaultdict(int)
    pl, pl_n = defaultdict(int), defaultdict(int)
    rew, comp = defaultdict(int), defaultdict(int)
    corr = defaultdict(int)

    for e in entries:
        wins[e["winner"]] += 1
        for agent, balance in e["final_balances"].items():
            if not _num(balance):
                continue
            runs[agent] += 1
            total_bal[agent] += balance
        for agent, s in (e.get("agent_stats") or {}).items():
            if not isinstance(s, dict):
                continue
            acc_c[agent] += s.get("correct") or 0
            corr[agent] += s.get("correct") or 0
            acc_g[agent] += s.get("graded_attempts") or 0
            if _num(s.get("net_pl")):
                pl[agent] += s["net_pl"]
                pl_n[agent] += 1
            rew[agent] += s.get("rewards_earned") or 0
            comp[agent] += s.get("compute_spent") or 0

    rows = []
    for agent, n in runs.items():
        rows.append({
            "agent": agent, "wins": wins.get(agent, 0), "runs": n,
            "win_rate": round(wins.get(agent, 0) / n, 3),
            "avg_final_balance": round(total_bal[agent] / n, 1),
            "avg_accuracy": safe_div(acc_c[agent], acc_g[agent]),
            "avg_net_pl": round(pl[agent] / pl_n[agent], 1) if pl_n[agent] else None,
            "efficiency": safe_div(rew[agent], comp[agent]),
            "cost_per_correct": safe_div(comp[agent], corr[agent], 2),
        })
    sign = 1 if sort_key in _ASCENDING else -1
    rows.sort(key=lambda r: (r[sort_key] is None, sign * (r[sort_key] or 0)))
    return jsonify({"total_runs": len(entries), "sorted_by": sort_key, "standings": rows,
                    "note": "win = highest final balance (wealth), not necessarily best problem solver"})


# --------------------------------------------------------------------- export

def _csv_safe(v) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str) if isinstance(v, (dict, list)) else \
        ("" if v is None else str(v))
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s  # CSV formula-injection guard


@app.route("/export/events.<fmt>")
def export_events(fmt):
    if fmt not in ("json", "csv"):
        return jsonify({"ok": False, "error": "format must be json or csv"}), 404
    all_events, err = _load_events(request.args.get("replay"))
    if err:
        return err
    if fmt == "json":
        return Response(json.dumps(all_events), mimetype="application/json",
                        headers={"Content-Disposition": "attachment; filename=events.json"})
    cols = []
    for e in all_events:
        for k in e:
            if k not in cols:
                cols.append(k)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for e in all_events:
        w.writerow([_csv_safe(e.get(c)) for c in cols])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=events.csv"})


@app.route("/export/summary.json")
def export_summary():
    all_events, err = _load_events(request.args.get("replay"))
    if err:
        return err
    st = build_state(all_events)
    summary = {k: st[k] for k in ("run_id", "mode", "seed", "sim_config", "models", "status",
                                  "ended_reason", "agents", "rankings", "stats", "gini", "tiers",
                                  "calibration", "provider_usage")}
    summary["note"] = ("All compute figures are simulated compute-seconds, not provider billing. "
                       "Mock-mode results describe the mock model, not a real LLM.")
    return Response(json.dumps(summary), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=summary.json"})


@app.route("/export/report.md")
def export_report():
    all_events, err = _load_events(request.args.get("replay"))
    if err:
        return err
    return Response(build_report(build_state(all_events)), mimetype="text/markdown",
                    headers={"Content-Disposition": "attachment; filename=report.md"})


@app.route("/events")
def events_page():
    """Paginated raw events: ?after_seq=N&limit=M (limit 1..500). Use next_after_seq to continue."""
    try:
        after = int(request.args.get("after_seq", 0))
        limit = int(request.args.get("limit", 100))
    except ValueError:
        return jsonify({"ok": False, "error": "after_seq and limit must be integers"}), 400
    if after < 0 or not 1 <= limit <= 500:
        return jsonify({"ok": False, "error": "after_seq must be >= 0 and limit between 1 and 500"}), 400
    all_events, err = _load_events(request.args.get("replay"))
    if err:
        return err
    page = [e for e in all_events if isinstance(e.get("seq"), int) and e["seq"] > after][:limit]
    return jsonify({"events": page, "count": len(page), "total": len(all_events),
                    "next_after_seq": page[-1]["seq"] if page else after})


@app.route("/experiments")
def experiments():
    out = []
    if EXPERIMENTS_DIR.exists():
        for d in sorted(EXPERIMENTS_DIR.iterdir()):
            f = d / "summary.json"
            if d.is_dir() and _NAME_RE.match(d.name) and f.exists():
                try:
                    s = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                out.append({"name": d.name, "kind": s.get("kind"), "param": s.get("param"), "mode": s.get("mode"),
                            "trials": s.get("trials"), "rounds": s.get("rounds")})
    return jsonify(out)


@app.route("/experiments/<name>")
def experiment_detail(name):
    if not _NAME_RE.match(name):
        return jsonify({"ok": False, "error": "invalid experiment name"}), 400
    d = EXPERIMENTS_DIR / name
    if not (d / "summary.json").exists() or d.resolve().parent != EXPERIMENTS_DIR.resolve():
        return jsonify({"ok": False, "error": "experiment not found"}), 404
    try:
        summary = json.loads((d / "summary.json").read_text())
    except (OSError, json.JSONDecodeError):
        return jsonify({"ok": False, "error": "experiment data unreadable"}), 500
    analysis = (d / "analysis.md").read_text() if (d / "analysis.md").exists() else ""
    return jsonify({"summary": summary, "analysis": analysis})


# --------------------------------------------------------------- inject task

@app.route("/inject_task", methods=["POST"])
def inject_task():
    """
    "Ask the economy": queue a viewer's question as a future round's task.
    Without an answer it is an ungraded exhibition round (no bounty). In live
    mode every queued question causes real model calls, so it is rate limited
    per client and globally, and refuses (429) instead of silently dropping
    when the queue is full.
    """
    if not config.ENABLE_INJECT:
        return jsonify({"ok": False, "error": "asking questions is disabled"}), 403
    body = _json_body()
    prompt, answer = body.get("prompt"), body.get("answer")
    if not isinstance(prompt, str):
        return jsonify({"ok": False, "error": "prompt is required"}), 400
    if answer is not None and not isinstance(answer, str):
        return jsonify({"ok": False, "error": "answer must be text"}), 400
    prompt = " ".join(prompt.split())
    answer = " ".join(answer.split()) if answer else None
    if not prompt:
        return jsonify({"ok": False, "error": "prompt is required"}), 400
    if len(prompt) > 500:
        return jsonify({"ok": False, "error": "prompt too long (500 char max)"}), 400
    if answer and len(answer) > 200:
        return jsonify({"ok": False, "error": "answer too long (200 char max)"}), 400

    ok, wait = limiter.allow(f"inject:{request.remote_addr}", config.INJECT_MAX_PER_MINUTE_PER_IP, 60)
    if ok:
        ok, wait = limiter.allow("inject:global", config.INJECT_MAX_PER_HOUR_GLOBAL, 3600)
    if not ok:
        resp = jsonify({"ok": False, "error": "rate limit exceeded, try again later"})
        resp.status_code = 429
        resp.headers["Retry-After"] = str(int(wait) + 1)
        return resp

    from core import tasks
    task = tasks.make_custom_task(prompt, answer=answer)
    if not injected_tasks.try_push(task):
        return jsonify({"ok": False, "error": "question queue is full, try again shortly"}), 429
    return jsonify({"ok": True, "queued_prompt": task.prompt, "graded": answer is not None,
                    "pending_ahead": injected_tasks.pending_count() - 1})


# ------------------------------------------------------------------- controls

@app.route("/control/pause", methods=["POST"])
@require_control
def control_pause():
    control.pause()
    return jsonify({"ok": True, "paused": True})


@app.route("/control/resume", methods=["POST"])
@require_control
def control_resume():
    control.resume()
    return jsonify({"ok": True, "paused": False})


@app.route("/control/speed", methods=["POST"])
@require_control
def control_speed():
    body = _json_body()
    if "seconds" not in body:
        raise _BadRequest("seconds is required")
    try:
        control.set_speed(body["seconds"])
    except ValueError as e:
        raise _BadRequest(str(e))
    return jsonify({"ok": True, "round_delay": control.round_delay})


@app.route("/control/save_replay", methods=["POST"])
@require_control
def control_save_replay():
    label = _json_body().get("label", "manual")
    if not isinstance(label, str):
        raise _BadRequest("label must be text")
    try:
        filename = events.save_replay(label=label)
    except OSError as e:
        log.error("save_replay failed: %s", redact(e))
        return jsonify({"ok": False, "error": "could not write replay file"}), 500
    return jsonify({"ok": True, "filename": filename})


@app.route("/control/reset", methods=["POST"])
@require_control
def control_reset():
    """Stop the current run and start a fresh one. Optional body:
    {"seed": int, "config": {num_agents, max_rounds, starting_balance, interest_rate, ...}}"""
    body = _json_body()
    overrides = body.get("config") or {}
    if not isinstance(overrides, dict):
        raise _BadRequest("config must be an object")
    if "seed" in body:
        overrides = {**overrides, "seed": body["seed"]}
    try:
        info = manager.start(overrides)
    except ConfigError as e:
        raise _BadRequest(str(e))
    return jsonify({"ok": True, **info})


def start_simulation_background():
    """Backward-compatible entry point; delegates to the lifecycle manager."""
    return manager.start()


# ----------------------------------------------------------------------- main

def _apply_proxy_fix():
    if config.TRUST_PROXY:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for w in config.config_warnings():
        log.warning("config: %s", w)
    mode = controls_mode()
    log.info("mode=%s controls=%s host=%s port=%s", config.mode(), mode, config.HOST, config.PORT)
    if config.mode() == "live" and not config.NEBIUS_API_KEY:
        raise SystemExit("USE_FAKE_MODEL=0 but NEBIUS_API_KEY is not set. "
                         "Set the key in .env, or use USE_FAKE_MODEL=1 for mock mode.")
    if mode == "open" and config.HOST not in ("127.0.0.1", "localhost", "::1"):
        log.warning("SECURITY: controls are OPEN and the server is bound to %s. Set "
                    "PUBLIC_MODE=1 and/or CONTROL_TOKEN before exposing this publicly.", config.HOST)
    _apply_proxy_fix()

    def _graceful(signum, _frame):
        log.info("signal %s: stopping simulation", signum)
        manager.shutdown(timeout=5.0)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _graceful)

    manager.start()
    try:
        from waitress import serve
        # one process, several threads: the simulation state is in-process
        serve(app, host=config.HOST, port=config.PORT, threads=8,
              max_request_body_size=config.MAX_REQUEST_BYTES)
    except ImportError:
        log.warning("waitress not installed; falling back to Flask's development server")
        app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
    finally:
        manager.shutdown(timeout=5.0)


if __name__ == "__main__":
    main()
