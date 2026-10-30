"""Dataset builder, fine-tune boundary, evaluation, provider check, experiments, config."""
import json
from pathlib import Path

import pytest

from core import config, tasks
from core.simconfig import ConfigError, SimulationConfig
from training import build_dataset as bd
from training import check_provider, evaluate, experiment, finetune

HELD = next(p for p, *_ in tasks._TASK_BANK if tasks.is_heldout(p))
TRAIN = next(p for p, *_ in tasks._TASK_BANK if not tasks.is_heldout(p))


def good_solve(prompt_text="q", answer="5"):
    return {"system": "s", "prompt": f"Question: {prompt_text}", "response": json.dumps({"answer": answer, "confidence": 0.9}),
            "tier": "nano", "purpose": "solve", "run_id": "r", "round": 1, "agent": "A", "task_id": 1}


def setup_run(isolated, calls, resolved_extra=None, task_prompt=TRAIN, injected=False):
    r = {"type": "task_resolved", "run_id": "r", "round": 1, "agent": "A", "correct": True, "tier": "nano",
         "bounty": 8, "cost": 1, "task_id": 1, "injected": injected, **(resolved_extra or {})}
    arrived = {"type": "task_arrived", "run_id": "r", "task_id": 1, "prompt": task_prompt, "injected": injected}
    bd.EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bd.EVENTS_PATH.write_text("\n".join(json.dumps(e) for e in (arrived, r)) + "\n")
    bd.CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bd.CALLS_PATH.write_text("\n".join(json.dumps(c) for c in calls) + "\n")


def build(**kw):
    return bd.build_datasets(min_examples=1, **kw)


def test_good_example_is_kept(isolated_data_dir):
    setup_run(isolated_data_dir, [good_solve(TRAIN)])
    m = build(strict=True)
    assert m["kept_nano"] == 1 and m["skipped"] == {}


@pytest.mark.parametrize("name,kwargs,reason", [
    ("heldout", dict(task_prompt=HELD), "heldout_task"),
    ("injected", dict(injected=True), "user_injected_task"),
    ("unprofitable", dict(resolved_extra={"tier": "ultra", "cost": 12, "bounty": 8}), "unprofitable"),
    ("wrong", dict(resolved_extra={"correct": False}), "bad_outcome"),
    ("exhibition", dict(resolved_extra={"correct": None}), "bad_outcome"),
])
def test_exclusions(isolated_data_dir, name, kwargs, reason):
    setup_run(isolated_data_dir, [good_solve(TRAIN)], **kwargs)
    m = build(strict=True)
    assert m["kept_nano"] == 0 and m["skipped"].get(reason) == 1


@pytest.mark.parametrize("response", ["not json", '{"answer": null}', '{"answer": ["x"]}', "", '{"x": 1}'])
def test_responses_the_app_would_reject_are_never_training_targets(isolated_data_dir, response):
    c = good_solve(TRAIN)
    c["response"] = response
    setup_run(isolated_data_dir, [c])
    m = build(strict=True)
    assert m["kept_nano"] == 0


def test_decision_examples_require_a_bid(isolated_data_dir):
    skip = {**good_solve(TRAIN), "purpose": "decision", "response": '{"action": "skip"}'}
    bid = {**good_solve(TRAIN), "purpose": "decision", "response": '{"action": "bid", "confidence": 0.9}'}
    setup_run(isolated_data_dir, [skip, bid])
    assert build(strict=True)["kept_nano"] == 1


def test_sensitive_content_duplicates_and_junk_records_dropped(isolated_data_dir):
    leak = good_solve(TRAIN)
    leak["prompt"] = "Question: my key is sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"
    email = {**good_solve(TRAIN), "prompt": "Question: mail bob@example.com"}
    ok = good_solve(TRAIN)
    setup_run(isolated_data_dir, [leak, email, ok, dict(ok), {"purpose": "solve"}, {"purpose": "epitaph"}])
    m = build(strict=True)
    assert m["kept_nano"] == 1
    assert m["skipped"]["sensitive_content"] == 2 and m["skipped"]["duplicate"] == 1


def test_unknown_provenance_dropped_only_in_strict_mode(isolated_data_dir):
    bd.EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bd.EVENTS_PATH.write_text(json.dumps({"type": "task_resolved", "run_id": "r", "round": 1, "agent": "A",
                                          "correct": True, "tier": "nano", "bounty": 8}) + "\n")
    bd.CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bd.CALLS_PATH.write_text(json.dumps(good_solve(TRAIN)) + "\n")
    assert build(strict=True)["skipped"]["unknown_provenance"] == 1
    assert build(strict=False)["kept_nano"] == 1


def test_raw_logs_are_never_modified_and_manifest_written(isolated_data_dir):
    setup_run(isolated_data_dir, [good_solve(TRAIN)])
    before = (bd.CALLS_PATH.read_bytes(), bd.EVENTS_PATH.read_bytes())
    build(strict=True)
    assert (bd.CALLS_PATH.read_bytes(), bd.EVENTS_PATH.read_bytes()) == before
    man = json.loads((bd.OUTPUT_DIR / "manifest.json").read_text())
    assert man["criteria"]["heldout_excluded"] and man["criteria"]["injected_excluded"]


def test_training_data_never_contains_a_heldout_prompt_end_to_end(isolated_data_dir):
    """Property check over the whole task bank."""
    calls, evs = [], []
    for i, (prompt, ans, _) in enumerate(tasks._TASK_BANK, 1):
        evs += [{"type": "task_arrived", "run_id": "r", "task_id": i, "prompt": prompt, "injected": False},
                {"type": "task_resolved", "run_id": "r", "round": i, "agent": "A", "correct": True, "tier": "nano",
                 "bounty": 8, "cost": 1, "task_id": i, "injected": False}]
        calls.append({**good_solve(prompt, ans), "round": i, "task_id": i})
    bd.EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bd.EVENTS_PATH.write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    bd.CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bd.CALLS_PATH.write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    build(strict=True)
    trained = (bd.OUTPUT_DIR / "nano_all.jsonl").read_text()
    for prompt, *_ in tasks._TASK_BANK:
        assert (f"Question: {prompt}" in trained) == (not tasks.is_heldout(prompt))


def test_no_calls_file_is_handled(isolated_data_dir):
    assert bd.build_datasets() is None


# ----------------------------------------------------------------- finetune
def write_ds(tmp, lines):
    p = tmp / "d.jsonl"
    p.write_text("\n".join(lines))
    return p


def test_validate_dataset_accepts_good_and_rejects_bad(tmp_path):
    ok = json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
    assert finetune.validate_dataset(write_ds(tmp_path, [ok, ok])) == 2
    bad_cases = ["not json", json.dumps({"messages": []}), json.dumps({"x": 1}), json.dumps([1]),
                 json.dumps({"messages": [{"role": "user", "content": "q"}]}),                      # no assistant last
                 json.dumps({"messages": [{"role": "bot", "content": "q"}, {"role": "assistant", "content": "a"}]}),
                 json.dumps({"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "a"}]}),
                 json.dumps({"messages": [{"role": "user", "content": 5}, {"role": "assistant", "content": "a"}]})]
    for bad in bad_cases:
        with pytest.raises(finetune.DatasetError):
            finetune.validate_dataset(write_ds(tmp_path, [bad]))
    with pytest.raises(finetune.DatasetError):
        finetune.validate_dataset(tmp_path / "missing.jsonl")
    with pytest.raises(finetune.DatasetError):
        finetune.validate_dataset(write_ds(tmp_path, [""]))


def test_job_request_shape():
    assert finetune.build_job_request("f1", "m", "s") == {"training_file": "f1", "model": "m", "suffix": "s"}
    assert finetune.build_job_request("f1", "m", "s", "f2")["validation_file"] == "f2"


class FakeFT:
    def __init__(self, final="succeeded"):
        self.final, self.created = final, None
        outer = self
        self.files = type("F", (), {"create": staticmethod(lambda file, purpose: type("O", (), {"id": "file-1"})())})()
        self.fine_tuning = type("T", (), {"jobs": type("J", (), {
            "create": staticmethod(lambda **kw: (setattr(outer, "created", kw), type("J", (), {"id": "job-1", "status": "queued"})())[1]),
            "retrieve": staticmethod(lambda jid: type("J", (), {"id": jid, "status": outer.final, "fine_tuned_model": "ft-x"})())})})()


def test_launch_dry_run_never_touches_network(tmp_path, capsys):
    ok = json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
    assert finetune.launch(write_ds(tmp_path, [ok]), "m", "s", dry_run=True) is None
    assert "not contacting Nebius" in capsys.readouterr().out


def test_launch_with_fake_client_and_deploy_note(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(finetune.time, "sleep", lambda s: None)
    ok = json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
    c = FakeFT()
    job = finetune.launch(write_ds(tmp_path, [ok]), "base", "sfx", dry_run=False, client=c)
    out = capsys.readouterr().out
    assert c.created == {"training_file": "file-1", "model": "base", "suffix": "sfx"} and job.status == "succeeded"
    assert "deploy" in out.lower() and "FINE_TUNED_NANO_MODEL" in out        # no false "paste this and done"


def test_failed_job_is_reported_not_celebrated(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(finetune.time, "sleep", lambda s: None)
    ok = json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
    finetune.launch(write_ds(tmp_path, [ok]), "b", "s", dry_run=False, client=FakeFT("failed"))
    out = capsys.readouterr().out
    assert "ended with status: failed" in out and "Job succeeded" not in out and "FINE_TUNED_NANO_MODEL" not in out


# ----------------------------------------------------------------- evaluate
def test_heldout_set_is_disjoint_from_training_split_and_nonempty():
    held = {t[0] for t in evaluate.heldout_tasks()}
    assert held and all(tasks.is_heldout(p) for p in held)


def test_evaluate_is_reproducible_and_stamped_mock():
    a, b = evaluate.evaluate(None, "x", seed=3), evaluate.evaluate(None, "x", seed=3)
    assert a == b and a["mode"] == "mock" and "Mock" in a["caveats"]
    lo, hi = a["accuracy_ci95"]
    assert lo <= a["accuracy"] <= hi


def test_compare_reports_overlap_and_never_claims_improvement():
    a, b = evaluate.evaluate(None, "base", seed=1), evaluate.evaluate(None, "tuned", seed=2)
    out = evaluate.compare(a, b)
    assert "overlap" in out and "MOCK" in out and "improv" not in out.lower()


# ------------------------------------------------------------ provider check
class FakeProvider:
    def __init__(self, models, content):
        self.models = type("M", (), {"list": staticmethod(lambda: type("L", (), {"data": [type("X", (), {"id": m})() for m in models]})())})()
        resp = type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})(), "finish_reason": "length" if content is None else "stop"})()]})()
        self.chat = type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(lambda **k: resp)})})()


def test_check_provider_pass_and_each_failure_mode(capsys):
    assert check_provider.run(FakeProvider([config.NANO_MODEL, config.ULTRA_MODEL], '{"answer": "4"}')) is True
    assert check_provider.run(FakeProvider(["other"], '{"answer": "4"}')) is False
    assert "FAIL" in capsys.readouterr().out
    assert check_provider.run(FakeProvider([config.NANO_MODEL, config.ULTRA_MODEL], None)) is False     # empty reasoning output
    assert check_provider.run(FakeProvider([config.NANO_MODEL, config.ULTRA_MODEL], "four")) is False


def test_check_provider_without_key_fails_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "")
    assert check_provider.run() is False and "NEBIUS_API_KEY" in capsys.readouterr().out


# --------------------------------------------------------------- experiments
def test_experiment_is_reproducible_isolated_and_reconciles(tmp_path, isolated_data_dir):
    args = ["--param", "interest_rate", "--values", "0,0.3", "--trials", "2", "--rounds", "10", "--seed", "50"]
    s1 = experiment.main(args + ["--out", str(tmp_path / "a")])
    s2 = experiment.main(args + ["--out", str(tmp_path / "b")])
    assert s1["conditions"] == s2["conditions"]
    assert all(c["ledger_all_ok"] for c in s1["conditions"].values())
    assert "fake model" in s1["warning"]
    assert (tmp_path / "a" / "events.jsonl").exists()      # experiment wrote only inside its own out dir


def test_experiment_rejects_invalid_conditions_and_live_without_confirmation(tmp_path, monkeypatch):
    with pytest.raises(ConfigError):
        experiment.main(["--param", "interest_rate", "--values", "5", "--out", str(tmp_path)])
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    with pytest.raises(SystemExit):
        experiment.main(["--param", "interest_rate", "--values", "0.1", "--out", str(tmp_path)])


# ------------------------------------------------------------------- config
@pytest.mark.parametrize("bad", [{"num_agents": 1}, {"num_agents": 6}, {"num_agents": "3"}, {"num_agents": True},
                                 {"interest_rate": -0.1}, {"interest_rate": float("nan")}, {"interest_rate": 2},
                                 {"seed": -1}, {"seed": 1.5}, {"max_rounds": 0}, {"max_rounds": 10**6},
                                 {"nano_cost": 5, "ultra_cost": 2}, {"escalation_cutoff": 1.5},
                                 {"task_weights": [0, 0, 0]}, {"task_weights": [1, 2]}, {"task_weights": [-1, 1, 1]},
                                 {"bogus": 1}])
def test_simconfig_rejects_invalid(bad):
    with pytest.raises(ConfigError):
        SimulationConfig.from_overrides(bad)


def test_simconfig_defaults_roundtrip():
    c = SimulationConfig.defaults(seed=5, task_weights=[1, 1, 0])
    assert c.rate_bp == 1500 and c.to_dict()["task_weights"] == [1.0, 1.0, 0.0]


def test_env_parsing_falls_back_and_warns(monkeypatch):
    import importlib
    monkeypatch.setenv("MODEL_MAX_TOKENS", "abc")
    monkeypatch.setenv("PORT", "99999")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "nan")
    importlib.reload(config)
    try:
        assert config.MODEL_MAX_TOKENS == 1024 and config.PORT == 5000 and config.MODEL_TIMEOUT_SECONDS == 20.0
        assert len(config.config_warnings()) >= 3
    finally:
        monkeypatch.undo()
        importlib.reload(config)
