"""Phase 6: provider usage capture, estimated cost, and 'never silently fake a real failure'."""
import threading
import types

import pytest

import core.orchestrator as orch
from core import config, events, model_client
from core.model_client import ModelCallError, estimate_cost_usd, take_usage


def resp(content="{}", usage="default"):
    if usage == "default":
        usage = types.SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150)
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content=content), finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=usage)


class Client:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def clean_usage():
    take_usage()
    yield
    take_usage()


def live_call(monkeypatch, *responses, tier="nano"):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(config, "LOG_TRAINING_DATA", False)
    monkeypatch.setattr(model_client, "_get_client", lambda: Client(*responses))
    return model_client.call_model("p", tier)


def test_usage_is_taken_from_the_provider_response_and_cleared_after_reading(monkeypatch):
    live_call(monkeypatch, resp())
    u = take_usage()
    assert (u["calls"], u["calls_with_usage"], u["input_tokens"], u["output_tokens"]) == (1, 1, 120, 30)
    assert u["by_tier"]["nano"]["model"] == config.NANO_MODEL
    assert take_usage() is None                                   # consumed


def test_usage_accumulates_per_tier_across_calls(monkeypatch):
    live_call(monkeypatch, resp(), tier="nano")
    live_call(monkeypatch, resp(usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)), tier="ultra")
    u = take_usage()
    assert u["calls"] == 2 and u["input_tokens"] == 130 and u["output_tokens"] == 35
    assert set(u["by_tier"]) == {"nano", "ultra"}


@pytest.mark.parametrize("bad_usage", [None, types.SimpleNamespace(), types.SimpleNamespace(prompt_tokens=None, completion_tokens=5),
                                       types.SimpleNamespace(prompt_tokens=-1, completion_tokens=5),
                                       types.SimpleNamespace(prompt_tokens="120", completion_tokens=5),
                                       types.SimpleNamespace(prompt_tokens=True, completion_tokens=5),
                                       types.SimpleNamespace(prompt_tokens=1.5, completion_tokens=5)])
def test_missing_or_malformed_usage_is_counted_as_a_call_but_never_as_tokens(monkeypatch, bad_usage):
    live_call(monkeypatch, resp(usage=bad_usage))
    u = take_usage()
    assert u["calls"] == 1 and u["calls_with_usage"] == 0 and u["input_tokens"] == 0 and u["output_tokens"] == 0


def test_failed_calls_record_no_usage(monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(model_client, "_get_client", lambda: Client(resp(content="")))
    with pytest.raises(ModelCallError):
        model_client.call_model("p", "nano")
    assert take_usage() is None


def test_mock_mode_never_reports_usage(isolated_data_dir):
    model_client.call_model("hello", "nano")
    assert take_usage() is None


def test_usage_is_isolated_per_thread(monkeypatch):
    live_call(monkeypatch, resp())
    seen = {}
    t = threading.Thread(target=lambda: seen.setdefault("other", take_usage()))
    t.start(); t.join()
    assert seen["other"] is None and take_usage()["calls"] == 1


# ------------------------------------------------------------- cost estimate
def test_cost_estimate_uses_only_prices_you_configured_and_never_guesses(monkeypatch):
    live_call(monkeypatch, resp(usage=types.SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=500_000)))
    u = take_usage()
    assert estimate_cost_usd(u) is None                                   # no prices configured -> None, not 0
    monkeypatch.setitem(config.PRICE_IN_PER_MTOK, "nano", 0.10)
    assert estimate_cost_usd(u) is None                                   # output price still missing
    monkeypatch.setitem(config.PRICE_OUT_PER_MTOK, "nano", 0.40)
    assert estimate_cost_usd(u) == pytest.approx(0.10 + 0.20)
    assert estimate_cost_usd(None) is None


def test_prices_are_not_hardcoded():
    import re, pathlib
    src = pathlib.Path(config.__file__).read_text()
    assert not re.search(r"PRICE_(IN|OUT)_PER_MTOK\s*=\s*\{[^}]*[0-9]\.[0-9]", src)


# ------------------------------------------------- integration with the run loop
def test_usage_events_are_emitted_per_round_in_live_mode_and_aggregate(isolated_data_dir, monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(config, "LOG_TRAINING_DATA", False)

    def fake_json(prompt, tier, system="", meta=None, model=None):
        model_client._record_usage(resp(), model or "m", tier)
        return {"action": "skip"} if "action" in system else {"answer": "x"}
    monkeypatch.setattr(orch, "call_model_json", fake_json)
    monkeypatch.setattr(orch, "_flavor_call", lambda *a, **k: None)
    events.reset_log()
    orch.run_simulation(max_rounds=3, round_delay=0, verbose=False, seed=1,
                        config_overrides={"strategies": "baseline"})
    us = [e for e in events.read_all() if e["type"] == "usage"]
    assert len(us) == 3 and all(e["usage"]["calls"] == 5 for e in us)         # 5 agents decide each round
    assert all(e["estimated_cost_usd"] is None for e in us)


def test_no_usage_events_in_mock_mode(isolated_data_dir):
    events.reset_log()
    orch.run_simulation(max_rounds=3, round_delay=0, verbose=False, seed=1)
    assert not [e for e in events.read_all() if e["type"] == "usage"]


# ---------------------------------------- failures are never masked by the fake
def test_a_real_failure_is_never_replaced_by_fake_output(isolated_data_dir, monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(model_client, "_get_client", lambda: Client(*[RuntimeError("provider down")] * 3))
    monkeypatch.setattr(model_client.time, "sleep", lambda s: None)

    def forbidden(*a, **k):
        raise AssertionError("the fake model must not be consulted when live mode is on")
    monkeypatch.setattr(model_client, "_fake_completion", forbidden)
    with pytest.raises(RuntimeError):
        model_client.call_model("p", "nano")                                 # surfaces, does not fall back


def test_live_mode_round_with_all_calls_failing_records_errors_not_fake_results(isolated_data_dir, monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)

    def down(*a, **k):
        raise ModelCallError("provider down", kind="server")
    monkeypatch.setattr(orch, "call_model_json", down)
    monkeypatch.setattr(orch, "_flavor_call", lambda *a, **k: None)
    events.reset_log()
    orch.run_simulation(max_rounds=2, round_delay=0, verbose=False, seed=1)
    evs = events.read_all()
    assert [e for e in evs if e["type"] == "call_error"]
    assert not [e for e in evs if e["type"] in ("task_resolved", "auction_resolved")]
    assert evs[-1]["type"] == "simulation_ended"
