"""HTTP API: routing, access control, input validation, limits, replays, exports."""
import json
import time

import pytest

import server
from core import config, events
from core.control import control
from core.injected_tasks import injected_tasks


@pytest.fixture
def client(isolated_data_dir):
    events.reset_log()
    return server.app.test_client()


# ------------------------------------------------------------------ routing
def test_every_documented_route_is_registered():
    rules = {(r.rule, m) for r in server.app.url_map.iter_rules() for m in r.methods}
    for path, method in [("/", "GET"), ("/dashboard.js", "GET"), ("/healthz", "GET"), ("/state", "GET"),
                         ("/replays", "GET"), ("/leaderboard", "GET"), ("/inject_task", "POST"),
                         ("/control/pause", "POST"), ("/control/resume", "POST"),
                         ("/control/speed", "POST"), ("/control/save_replay", "POST"),
                         ("/control/reset", "POST")]:
        assert (path, method) in rules, f"{method} {path} is not routed"


def test_pause_and_resume_actually_toggle_state(client):
    assert client.post("/control/pause").json == {"ok": True, "paused": True}
    assert control.is_paused() and client.get("/state").json["paused"] is True
    assert client.post("/control/resume").json == {"ok": True, "paused": False}
    assert not control.is_paused()


def test_static_files_only_the_two_expected_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/dashboard.js").status_code == 200
    for p in ("/web/dashboard.html", "/server.py", "/.env", "/data/events.jsonl", "/static/x", "/core/config.py"):
        assert client.get(p).status_code == 404


def test_healthz_reports_mode_and_never_secrets(client, monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "sk-supersecretkey-123456")
    body = client.get("/healthz").get_data(as_text=True)
    assert "supersecret" not in body and client.get("/healthz").json["mode"] == "mock"


# ------------------------------------------------------------ access control
def test_open_mode_allows_controls(client):
    assert client.post("/control/pause").status_code == 200


def test_public_mode_without_token_fails_closed(client, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_MODE", True)
    monkeypatch.setattr(config, "CONTROL_TOKEN", "")
    for path in ("/control/pause", "/control/resume", "/control/speed", "/control/save_replay", "/control/reset"):
        assert client.post(path, json={}).status_code == 403
    assert not control.is_paused()
    assert client.get("/state").json["controls"] == "disabled"
    assert client.get("/state").status_code == 200              # reading still works


def test_token_required_and_checked(client, monkeypatch):
    monkeypatch.setattr(config, "CONTROL_TOKEN", "s3cret-token")
    assert client.post("/control/pause").status_code == 401
    assert client.post("/control/pause", headers={"X-Control-Token": "wrong"}).status_code == 401
    assert client.post("/control/pause", headers={"X-Control-Token": ""}).status_code == 401
    assert not control.is_paused()
    assert client.post("/control/pause", headers={"X-Control-Token": "s3cret-token"}).status_code == 200
    assert control.is_paused()
    client.post("/control/resume", headers={"X-Control-Token": "s3cret-token"})


def test_token_protects_reset_even_in_public_mode(client, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_MODE", True)
    monkeypatch.setattr(config, "CONTROL_TOKEN", "tok")
    assert client.post("/control/reset", json={}).status_code == 401


def test_inject_remains_available_to_viewers_in_public_mode(client, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_MODE", True)
    assert client.post("/inject_task", json={"prompt": "hello?"}).status_code == 200


def test_inject_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_INJECT", False)
    assert client.post("/inject_task", json={"prompt": "hi"}).status_code == 403


# --------------------------------------------------------------- validation
@pytest.mark.parametrize("body", [{"seconds": "abc"}, {"seconds": None}, {"seconds": []}, {}, {"seconds": "nan"},
                                  {"seconds": {"a": 1}}])
def test_speed_rejects_garbage(client, body):
    before = control.round_delay
    r = client.post("/control/speed", json=body)
    assert r.status_code == 400 and control.round_delay == before


def test_speed_clamps_valid_values(client):
    assert client.post("/control/speed", json={"seconds": 1000}).json["round_delay"] == 30.0
    assert client.post("/control/speed", json={"seconds": 0.001}).json["round_delay"] == 0.1


@pytest.mark.parametrize("raw", ["not json", "[1,2]", '"str"', "123"])
def test_non_object_json_bodies_are_400_not_500(client, raw):
    r = client.post("/control/speed", data=raw, content_type="application/json")
    assert r.status_code == 400


@pytest.mark.parametrize("body", [{"prompt": 5}, {"prompt": None}, {"prompt": ""}, {"prompt": "   "},
                                  {"prompt": ["x"]}, {"prompt": "x" * 501}, {"prompt": "ok", "answer": 5},
                                  {"prompt": "ok", "answer": "x" * 201}, {}])
def test_inject_validation(client, body):
    r = client.post("/inject_task", json=body)
    assert r.status_code == 400 and injected_tasks.pending_count() == 0


def test_inject_normalizes_whitespace_and_newlines(client):
    r = client.post("/inject_task", json={"prompt": "line one\n\nline two\t!"}).json
    assert r["queued_prompt"] == "line one line two !"


def test_html_in_prompt_is_stored_verbatim_and_returned_only_as_json(client):
    payload = "<img src=x onerror=alert(1)><script>alert(2)</script>"
    r = client.post("/inject_task", json={"prompt": payload})
    assert r.status_code == 200 and r.mimetype == "application/json"
    assert r.json["queued_prompt"] == payload                  # not mangled, not rendered


def test_request_body_size_limit(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_REQUEST_BYTES", 300)
    r = client.post("/inject_task", json={"prompt": "x" * 400})
    assert r.status_code == 413


def test_inject_per_ip_rate_limit_then_recovers_per_other_ip(client, monkeypatch):
    monkeypatch.setattr(config, "INJECT_MAX_PER_MINUTE_PER_IP", 3)
    codes = [client.post("/inject_task", json={"prompt": f"q{i}"}).status_code for i in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = client.post("/inject_task", json={"prompt": "q"})
    assert r.headers["Retry-After"].isdigit()
    other = client.post("/inject_task", json={"prompt": "q"}, environ_overrides={"REMOTE_ADDR": "9.9.9.9"})
    assert other.status_code == 200


def test_inject_global_hourly_cap(client, monkeypatch):
    monkeypatch.setattr(config, "INJECT_MAX_PER_HOUR_GLOBAL", 2)
    monkeypatch.setattr(config, "INJECT_MAX_PER_MINUTE_PER_IP", 100)
    codes = [client.post("/inject_task", json={"prompt": "q"}, environ_overrides={"REMOTE_ADDR": f"1.1.1.{i}"}).status_code
             for i in range(4)]
    assert codes == [200, 200, 429, 429]


def test_full_queue_rejects_instead_of_silently_dropping(client, monkeypatch):
    monkeypatch.setattr(config, "INJECT_MAX_PER_MINUTE_PER_IP", 1000)
    n = injected_tasks._queue.maxlen
    for i in range(n):
        assert client.post("/inject_task", json={"prompt": f"q{i}"}).status_code == 200
    r = client.post("/inject_task", json={"prompt": "overflow"})
    assert r.status_code == 429 and injected_tasks.pending_count() == n
    assert injected_tasks.pop_next().prompt == "q0"             # oldest was NOT evicted


# ---------------------------------------------------------------- reset API
def test_reset_validates_config_and_seed(client):
    assert client.post("/control/reset", json={"config": {"num_agents": 99}}).status_code == 400
    assert client.post("/control/reset", json={"config": {"bogus": 1}}).status_code == 400
    assert client.post("/control/reset", json={"config": "x"}).status_code == 400
    assert client.post("/control/reset", json={"seed": "abc"}).status_code == 400
    assert client.post("/control/reset", json={"seed": -5}).status_code == 400
    r = client.post("/control/reset", json={"seed": 7, "config": {"max_rounds": 3}})
    assert r.status_code == 200 and r.json["run_id"]
    control.round_delay = 0


def test_reset_starts_clean_run_visible_immediately(client):
    client.post("/control/reset", json={"seed": 1, "config": {"max_rounds": 2}})
    st = client.get("/state").json
    assert st["run_id"] and st["round_count"] <= 1


# ------------------------------------------------------------------ replays
def test_replay_traversal_missing_and_valid(client):
    assert client.get("/state?replay=../../etc/passwd").status_code == 400
    assert client.get("/state?replay=a/b.jsonl").status_code == 400
    assert client.get("/state?replay=missing.jsonl").status_code == 404
    events.emit("task_arrived", round=1, task_id=1, prompt="p", difficulty="easy", bounty=8)
    name = events.save_replay("t")
    r = client.get(f"/state?replay={name}")
    assert r.status_code == 200 and r.json["replay"] is True and r.json["status"] == "replay"


def test_replay_view_is_read_only_and_survives_corruption(client):
    events.emit("task_arrived", round=1, task_id=1, prompt="p", difficulty="easy", bounty=8)
    name = events.save_replay("t")
    path = events.REPLAY_DIR / name
    with open(path, "a") as f:
        f.write("{corrupt\n[]\n{}\n")
    before = path.read_bytes()
    live_before = events.LOG_PATH.read_bytes()
    r = client.get(f"/state?replay={name}")
    assert r.status_code == 200
    assert path.read_bytes() == before and events.LOG_PATH.read_bytes() == live_before
    assert any(x["skipped_lines"] >= 2 for x in client.get("/replays").json)


def test_corrupt_live_log_never_500s(client):
    with open(events.LOG_PATH, "a") as f:
        f.write('{"type": "decision", "agent"\nnull\n{"no_type": 1}\n{"type": 5}\n')
    assert client.get("/state").status_code == 200


# --------------------------------------------------------------- leaderboard
def test_leaderboard_handles_legacy_and_garbage_entries(client):
    rows = [
        {"winner": "A", "final_balances": {"A": 150, "B": 50}},                              # legacy (no agent_stats)
        {"winner": "B", "final_balances": {"A": 90, "B": 110}, "agent_stats": {
            "A": {"correct": 3, "graded_attempts": 4, "net_pl": -10, "rewards_earned": 24, "compute_spent": 6},
            "B": {"correct": 1, "graded_attempts": 4, "net_pl": 10, "rewards_earned": 8, "compute_spent": 4}}},
        {"winner": 5, "final_balances": "x"}, {"winner": "A"}, {"final_balances": {}},          # garbage
        {"winner": "A", "final_balances": {"A": "NaN", "B": True}},
    ]
    with open(events.LEADERBOARD_PATH, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    r = client.get("/leaderboard")
    assert r.status_code == 200
    by = {s["agent"]: s for s in r.json["standings"]}
    assert by["A"]["runs"] == 2 and by["A"]["wins"] == 1 and by["A"]["avg_accuracy"] == 0.75
    assert by["B"]["avg_net_pl"] == 10.0
    assert client.get("/leaderboard?sort=accuracy").json["standings"][0]["agent"] == "A"
    assert client.get("/leaderboard?sort=evil").status_code == 400


def test_leaderboard_empty(client):
    assert client.get("/leaderboard").json["standings"] == []


# -------------------------------------------------------------------- export
def test_csv_export_neutralizes_formula_injection(client):
    events.emit("note", text="=HYPERLINK(\"http://evil\",\"x\")", other="+cmd", ok="fine")
    body = client.get("/export/events.csv").get_data(as_text=True)
    assert "'=HYPERLINK" in body and "'+cmd" in body and ",fine" in body


def test_export_json_and_summary_and_bad_format(client):
    events.emit("task_arrived", round=1, task_id=1, prompt="p", difficulty="easy", bounty=8)
    assert client.get("/export/events.json").json[0]["type"] == "task_arrived"
    s = client.get("/export/summary.json").json
    assert "simulated" in s["note"] and "stats" in s
    assert client.get("/export/events.xml").status_code == 404


# ------------------------------------------------------------ error handling
def test_unhandled_errors_return_generic_json_without_internals(client, monkeypatch):
    monkeypatch.setattr(server, "build_state", lambda e: (_ for _ in ()).throw(RuntimeError("SECRET /home/x/db.py")))
    r = client.get("/state")
    assert r.status_code == 500 and r.json == {"ok": False, "error": "internal server error"}
    assert "SECRET" not in r.get_data(as_text=True) and "Traceback" not in r.get_data(as_text=True)


def test_http_errors_are_json(client):
    assert client.get("/nope").json == {"ok": False, "error": "not found"}
    r = client.get("/control/pause")                            # wrong method
    assert r.status_code == 405 and r.is_json


def test_security_headers_present(client):
    h = client.get("/state").headers
    assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "DENY"
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-eval" not in csp and "frame-ancestors 'none'" in csp
    assert "script-src 'self' 'unsafe-inline'" not in csp
    assert "Access-Control-Allow-Origin" not in h               # no CORS: same-origin only
    assert client.get("/state").headers["Cache-Control"] == "no-store"


def test_state_exposes_mode_and_is_honest_in_mock(client):
    events.emit("simulation_started", agents=["A"], starting_balance=100, mode="mock", seed=1,
                config={}, models={"nano": "n", "ultra": "u"})
    st = client.get("/state").json
    assert st["mode"] == "mock" and st["seed"] == 1
