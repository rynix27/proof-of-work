"""Report generation, event pagination and experiment browsing."""
import json

import pytest

import core.orchestrator as orch
import server
from core import events
from core.report import build_report
from core.state import build_state


@pytest.fixture
def client(isolated_data_dir):
    events.reset_log()
    return server.app.test_client()


@pytest.fixture
def finished_run(client):
    orch.run_simulation(max_rounds=30, round_delay=0, verbose=False, seed=8)
    return client


# ------------------------------------------------------------------- report
def test_report_is_generated_from_the_same_numbers_as_the_dashboard(finished_run):
    st = build_state(events.read_all())
    md = finished_run.get("/export/report.md").get_data(as_text=True)
    s = st["stats"]
    assert f"economic profit: **{s['economic_profit']}**" in md
    assert f"{s['tasks_correct']}/{s['tasks_resolved']}" in md
    for name, a in st["agents"].items():
        assert f"| {name}" in md and a["strategy"] in md
    assert st["run_id"] in md and f"`{st['seed']}`" in md


def test_mock_report_is_labelled_mock_and_states_its_limits(finished_run):
    md = finished_run.get("/export/report.md").get_data(as_text=True)
    assert md.splitlines()[2].startswith("> **MOCK RUN.**")
    for phrase in ("not money", "Single run", "self-reported confidence", "does not depend on tier or strategy",
                   "mock mode makes no provider calls"):
        assert phrase in md


def test_report_headers_and_replay_support(finished_run):
    r = finished_run.get("/export/report.md")
    assert r.mimetype == "text/markdown" and "attachment" in r.headers["Content-Disposition"]
    name = events.save_replay("t")
    assert finished_run.get(f"/export/report.md?replay={name}").status_code == 200
    assert finished_run.get("/export/report.md?replay=../x.jsonl").status_code == 400
    assert finished_run.get("/export/report.md?replay=nope.jsonl").status_code == 404


def test_report_on_an_empty_log_does_not_crash_or_invent_numbers(client):
    md = client.get("/export/report.md").get_data(as_text=True)
    assert "n/a" in md and "Status: **idle**" in md


def test_report_never_presents_a_made_up_cost():
    st = build_state([])
    md = build_report(st)
    assert "$" not in md                                              # no usage -> no dollar figure
    st["provider_usage"] = {"calls": 10, "calls_with_usage": 8, "input_tokens": 100, "output_tokens": 50,
                            "estimated_cost_usd": None}
    assert "unknown (prices not configured)" in build_report(st) and "$" not in build_report(st)
    st["provider_usage"]["estimated_cost_usd"] = 0.0123
    assert "$0.0123" in build_report(st)


def test_report_flags_small_escalation_samples(finished_run):
    assert "Smallest tier sample" in finished_run.get("/export/report.md").get_data(as_text=True)


def test_summary_export_includes_the_new_sections(finished_run):
    s = finished_run.get("/export/summary.json").json
    assert {"tiers", "calibration", "provider_usage"} <= set(s) and "simulated" in s["note"]


# -------------------------------------------------------------- pagination
def test_events_pagination_walks_the_whole_log_without_gaps_or_repeats(finished_run):
    total = len(events.read_all())
    seen, after = [], 0
    while True:
        r = finished_run.get(f"/events?after_seq={after}&limit=97").json
        if not r["events"]:
            break
        seen += [e["seq"] for e in r["events"]]
        assert r["count"] == len(r["events"]) <= 97 and r["total"] == total
        after = r["next_after_seq"]
    assert seen == sorted(seen) == list(range(1, total + 1))


@pytest.mark.parametrize("qs", ["after_seq=-1", "limit=0", "limit=501", "limit=abc", "after_seq=1.5", "limit=-3"])
def test_events_pagination_rejects_bad_parameters(client, qs):
    assert client.get(f"/events?{qs}").status_code == 400


def test_events_pagination_end_state_and_replay(finished_run):
    last = finished_run.get("/events?after_seq=0&limit=500").json["next_after_seq"]
    r = finished_run.get(f"/events?after_seq={10**9}").json
    assert r["events"] == [] and r["next_after_seq"] == 10**9
    name = events.save_replay("p")
    assert finished_run.get(f"/events?replay={name}&limit=5").json["count"] == 5
    assert last > 0


def test_events_endpoint_tolerates_events_without_seq(client):
    with open(events.LOG_PATH, "a") as f:
        f.write(json.dumps({"type": "legacy_event_without_seq"}) + "\n")
    events.emit("modern")
    r = client.get("/events").json
    assert [e["type"] for e in r["events"]] == ["modern"]            # legacy rows are skipped, not a crash


# ------------------------------------------------------------- experiments
@pytest.fixture
def exp_dir(tmp_path, monkeypatch):
    d = tmp_path / "experiments"
    monkeypatch.setattr(server, "EXPERIMENTS_DIR", d)
    return d


def make_experiment(root, name, **summary):
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"kind": "sweep", "param": "interest_rate", "mode": "mock",
                                                "trials": 5, "rounds": 20, **summary}))
    (d / "analysis.md").write_text("# analysis")
    return d


def test_experiments_listing_and_detail(client, exp_dir):
    assert client.get("/experiments").json == []                      # no directory yet
    make_experiment(exp_dir, "interest-1")
    make_experiment(exp_dir, "bad", )
    (exp_dir / "bad" / "summary.json").write_text("{not json")        # corrupt entries are skipped
    (exp_dir / "stray.txt").write_text("x")
    (exp_dir / "nosummary").mkdir()
    listing = client.get("/experiments").json
    assert [e["name"] for e in listing] == ["interest-1"] and listing[0]["trials"] == 5
    d = client.get("/experiments/interest-1").json
    assert d["summary"]["param"] == "interest_rate" and d["analysis"] == "# analysis"


@pytest.mark.parametrize("name", ["../etc", "..", "a/b", ".hidden", "x" * 61, "a b", "a;b", "%2e%2e"])
def test_experiment_names_are_validated(client, exp_dir, name):
    r = client.get(f"/experiments/{name}")
    assert r.status_code in (400, 404)
    assert "root:" not in r.get_data(as_text=True)


def test_missing_and_unreadable_experiments(client, exp_dir):
    assert client.get("/experiments/missing").status_code == 404
    make_experiment(exp_dir, "broken")
    (exp_dir / "broken" / "summary.json").write_text("{not json")
    assert client.get("/experiments/broken").status_code == 500
    assert "Traceback" not in client.get("/experiments/broken").get_data(as_text=True)


def test_new_read_endpoints_are_rate_limited(client, monkeypatch):
    from core import config
    monkeypatch.setattr(config, "READ_MAX_PER_MINUTE", 10)
    assert [client.get("/events").status_code for _ in range(12)].count(429) == 2


# ------------------------------------------------------ leaderboard: cost per correct
def test_leaderboard_cost_per_correct_ranks_lowest_cost_first_and_none_last(client):
    def row(winner, a_stats, b_stats):
        return {"winner": winner, "final_balances": {"A": 100, "B": 100, "C": 100},
                "agent_stats": {"A": a_stats, "B": b_stats, "C": {"correct": 0, "compute_spent": 5, "graded_attempts": 5}}}
    rows = [row("A", {"correct": 10, "compute_spent": 20, "graded_attempts": 12},
                {"correct": 5, "compute_spent": 40, "graded_attempts": 8})]
    with open(events.LEADERBOARD_PATH, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    r = client.get("/leaderboard?sort=cost").json
    order = [(s["agent"], s["cost_per_correct"]) for s in r["standings"]]
    assert order == [("A", 2.0), ("B", 8.0), ("C", None)]          # 20/10, 40/5, and 0 correct -> unknown, last
    assert r["sorted_by"] == "cost_per_correct"
    assert client.get("/leaderboard?sort=accuracy").json["standings"][0]["agent"] in ("A", "B", "C")
