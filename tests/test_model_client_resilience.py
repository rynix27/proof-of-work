"""Provider failure handling, charging classification, bounds, and mock determinism."""
import openai
import pytest

from core import config, model_client
from core.model_client import ModelCallError
from tests._compat import httpx


def _req():
    return httpx.Request("POST", "https://x/v1/chat/completions")


class Msg:
    def __init__(self, content, finish="stop"):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})(), "finish_reason": finish})()]


class Client:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0
        outer = self

        class Completions:
            @staticmethod
            def create(**kwargs):
                outer.calls += 1
                outer.kwargs = kwargs
                o = outer.outcomes.pop(0)
                if isinstance(o, Exception):
                    raise o
                return o
        self.chat = type("Chat", (), {"completions": Completions})()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(model_client.time, "sleep", lambda s: None)


def status_err(code):
    cls = {429: openai.RateLimitError, 401: openai.AuthenticationError, 404: openai.NotFoundError,
           400: openai.BadRequestError, 500: openai.InternalServerError, 503: openai.InternalServerError}[code]
    return cls("boom", response=httpx.Response(code, request=_req()), body=None)


@pytest.mark.parametrize("code,kind", [(401, "auth"), (404, "rejected"), (400, "rejected")])
def test_4xx_fails_fast_uncharged(code, kind):
    c = Client(status_err(code))
    with pytest.raises(ModelCallError) as e:
        model_client._call_with_retry(c, "m", [], "nano")
    assert (e.value.kind, e.value.charged, c.calls) == (kind, False, 1)


@pytest.mark.parametrize("code", [500, 503])
def test_5xx_is_retried_then_succeeds(code):
    c = Client(status_err(code), status_err(code), Msg('{"a":1}'))
    assert model_client._call_with_retry(c, "m", [], "nano") == '{"a":1}' and c.calls == 3


def test_5xx_exhaustion_is_uncharged():
    c = Client(*[status_err(503)] * 3)
    with pytest.raises(ModelCallError) as e:
        model_client._call_with_retry(c, "m", [], "nano")
    assert (e.value.kind, e.value.charged, c.calls) == ("server", False, 3)


def test_rate_limit_exhaustion_uncharged_timeout_exhaustion_charged_once():
    c = Client(*[status_err(429)] * 3)
    with pytest.raises(ModelCallError) as e:
        model_client._call_with_retry(c, "m", [], "nano")
    assert (e.value.kind, e.value.charged) == ("rate_limit", False)
    c = Client(*[openai.APITimeoutError(request=_req())] * 3)
    with pytest.raises(ModelCallError) as e:
        model_client._call_with_retry(c, "m", [], "nano")
    assert (e.value.kind, e.value.charged, c.calls) == ("timeout", True, 3)   # one charge, three attempts


def test_connection_error_is_uncharged():
    c = Client(*[openai.APIConnectionError(request=_req())] * 3)
    with pytest.raises(ModelCallError) as e:
        model_client._call_with_retry(c, "m", [], "nano")
    assert (e.value.kind, e.value.charged) == ("connection", False)


@pytest.mark.parametrize("content", [None, "", "   ", "<think>only reasoning</think>"])
def test_empty_content_is_an_error_with_actionable_hint(content):
    c = Client(Msg(content, finish="length"))
    with pytest.raises(ModelCallError) as e:
        model_client._call_with_retry(c, "m", [], "nano")
    assert e.value.kind == "empty" and e.value.charged is True and "MODEL_MAX_TOKENS" in str(e.value)


def test_malformed_response_shape_is_handled():
    c = Client(type("R", (), {"choices": []})())
    with pytest.raises(ModelCallError) as e:
        model_client._call_with_retry(c, "m", [], "nano")
    assert e.value.kind == "empty"


def test_wall_clock_deadline_stops_retries(monkeypatch):
    monkeypatch.setattr(config, "MODEL_DEADLINE_SECONDS", 0.5)
    c = Client(*[status_err(429)] * 8)          # backoff 2s,4s,... exceeds the 0.5s deadline
    with pytest.raises(ModelCallError):
        model_client._call_with_retry(c, "m", [], "nano", max_retries=8)
    assert c.calls == 1


def test_request_uses_configured_token_budget_and_timeout(monkeypatch):
    monkeypatch.setattr(config, "MODEL_MAX_TOKENS", 777)
    c = Client(Msg("ok"))
    model_client._call_with_retry(c, "m", [], "nano", timeout_seconds=9)
    assert c.kwargs["max_tokens"] == 777 and c.kwargs["timeout"] == 9


def test_sdk_client_built_without_its_own_retries(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(model_client, "_client", None)
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "k" * 12)
    model_client._get_client()
    assert seen["max_retries"] == 0 and seen["base_url"].startswith("https://api.tokenfactory.nebius.com")
    monkeypatch.setattr(model_client, "_client", None)


def test_missing_key_gives_clear_config_error(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "")
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(model_client, "_client", None)
    with pytest.raises(ModelCallError) as e:
        model_client.call_model("hi", "nano")
    assert e.value.kind == "config" and "NEBIUS_API_KEY" in str(e.value) and "USE_FAKE_MODEL" in str(e.value)


def test_call_model_json_returns_parse_error_marker_and_strips_think(monkeypatch):
    monkeypatch.setattr(model_client, "call_model", lambda *a, **k: ("<think>x</think>```json\n{\"a\": 1}\n```", 0.1))
    assert model_client.call_model_json("p", "nano")["a"] == 1
    monkeypatch.setattr(model_client, "call_model", lambda *a, **k: ("garbage", 0.1))
    r = model_client.call_model_json("p", "nano")
    assert r["_parse_error"] is True and r["_error"] == "malformed_json"


def test_fake_model_is_deterministic_per_seed_and_never_logged(isolated_data_dir):
    def sample(seed):
        model_client.seed_fake(seed)
        return [model_client.call_model('{"action": x}', "nano")[0] for _ in range(8)]
    assert sample(1) == sample(1) and sample(1) != sample(2)
    assert not model_client.TRAINING_LOG_PATH.exists()


def test_model_override_beats_fine_tuned_beats_base(monkeypatch):
    monkeypatch.setattr(config, "FINE_TUNED_NANO_MODEL", "")
    assert model_client._resolve_model("nano") == config.NANO_MODEL
    monkeypatch.setattr(config, "FINE_TUNED_NANO_MODEL", "ft")
    assert model_client._resolve_model("nano") == "ft"
    # explicit model= bypasses resolution in call_model (live path)
    seen = {}
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(model_client, "_get_client", lambda: object())
    monkeypatch.setattr(model_client, "_call_with_retry", lambda c, m, msgs, t, **k: seen.setdefault("m", m) and "{}")
    monkeypatch.setattr(config, "LOG_TRAINING_DATA", False)
    model_client.call_model("p", "nano", model="explicit-model")
    assert seen["m"] == "explicit-model"


def test_live_calls_are_logged_with_actual_model_and_injected_flag(isolated_data_dir, monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(model_client, "_get_client", lambda: object())
    monkeypatch.setattr(model_client, "_call_with_retry", lambda *a, **k: '{"answer": "1"}')
    model_client.call_model("p", "nano", system="s", meta={"purpose": "solve", "injected": True}, model="mm")
    import json
    rec = json.loads(model_client.TRAINING_LOG_PATH.read_text().splitlines()[0])
    assert rec["model"] == "mm" and rec["injected"] is True and rec["purpose"] == "solve"
