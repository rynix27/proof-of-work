"""
Tests for core/model_client.py -- focused on the retry/resilience logic,
since that's the part protecting live demos from a single flaky API call
taking down the whole simulation. The fake-mode gating (real calls only
logged when not in fake mode) is also covered here.
"""
import json

import openai

from tests._compat import httpx as httpx2
import pytest

from core import config, model_client


def _fake_request():
    return httpx2.Request("POST", "https://example.com/v1/chat/completions")


class _CountingClient:
    """Stands in for the OpenAI client; each call raises via `behavior`
    until exhausted, then returns a real-looking response."""

    def __init__(self, behavior):
        self.behavior = list(behavior)
        self.call_count = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.call_count += 1
        if self.behavior:
            action = self.behavior.pop(0)
            if action is not None:
                raise action
        return _FakeResponse("ok response")


class _FakeResponse:
    def __init__(self, text):
        self.choices = [_FakeChoice(text)]


class _FakeChoice:
    def __init__(self, text):
        self.message = _FakeMessage(text)


class _FakeMessage:
    def __init__(self, text):
        self.content = text


def test_rate_limit_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(model_client.time, "sleep", lambda s: None)
    req = _fake_request()
    resp = httpx2.Response(429, request=req)
    err = openai.RateLimitError("rate limited", response=resp, body=None)

    client = _CountingClient([err, err, None])  # fails twice, succeeds third try
    result = model_client._call_with_retry(client, "some-model", [{"role": "user", "content": "hi"}], "nano")

    assert result == "ok response"
    assert client.call_count == 3


def test_timeout_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(model_client.time, "sleep", lambda s: None)
    req = _fake_request()
    err = openai.APITimeoutError(request=req)

    client = _CountingClient([err, None])
    result = model_client._call_with_retry(client, "some-model", [{"role": "user", "content": "hi"}], "nano")

    assert result == "ok response"
    assert client.call_count == 2


def test_exhausted_retries_raises_clear_error(monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(model_client.time, "sleep", lambda s: None)
    req = _fake_request()
    resp = httpx2.Response(429, request=req)
    err = openai.RateLimitError("rate limited", response=resp, body=None)

    client = _CountingClient([err, err, err, err, err])  # never succeeds
    with pytest.raises(RuntimeError, match="failed after"):
        model_client._call_with_retry(client, "some-model", [{"role": "user", "content": "hi"}], "nano",
                                       max_retries=3)
    assert client.call_count == 3  # doesn't retry forever


def test_bad_status_fails_fast_without_retrying(monkeypatch):
    """A 4xx that isn't a rate limit (bad model name, bad auth) should not
    burn through retries -- it fails immediately with a clear message."""
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    req = _fake_request()
    resp = httpx2.Response(401, request=req)
    err = openai.APIStatusError("invalid api key", response=resp, body=None)

    client = _CountingClient([err])
    with pytest.raises(RuntimeError, match="NEBIUS_API_KEY"):
        model_client._call_with_retry(client, "some-model", [{"role": "user", "content": "hi"}], "nano")
    assert client.call_count == 1  # failed fast, no retries wasted


def test_fake_mode_calls_are_never_logged_as_training_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", True)
    monkeypatch.setattr(config, "LOG_TRAINING_DATA", True)
    monkeypatch.setattr(model_client, "TRAINING_LOG_PATH", tmp_path / "calls.jsonl")

    model_client.call_model("Question: what is 2+2?", tier="nano",
                            system='Respond with {"answer": "...", "confidence": 0.5}')

    assert not (tmp_path / "calls.jsonl").exists(), \
        "fake-mode calls must never be logged -- there's nothing genuine to learn from"


def test_real_mode_calls_are_logged_with_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(config, "LOG_TRAINING_DATA", True)
    monkeypatch.setattr(model_client, "TRAINING_LOG_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(model_client, "_get_client", lambda: _CountingClient([None]))

    model_client.call_model("Question: what is 2+2?", tier="nano", system="answer briefly",
                            meta={"agent": "Specialist", "round": 3, "purpose": "solve"})

    lines = (tmp_path / "calls.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["agent"] == "Specialist"
    assert record["round"] == 3
    assert record["purpose"] == "solve"
    assert record["response"] == "ok response"


def test_fine_tuned_model_override_takes_priority(monkeypatch):
    monkeypatch.setattr(config, "FINE_TUNED_NANO_MODEL", "my-custom-nano-v1")
    monkeypatch.setattr(config, "NANO_MODEL", "nvidia/nemotron-nano")
    assert model_client._resolve_model("nano") == "my-custom-nano-v1"


def test_no_override_falls_back_to_base_model(monkeypatch):
    monkeypatch.setattr(config, "FINE_TUNED_NANO_MODEL", "")
    monkeypatch.setattr(config, "NANO_MODEL", "nvidia/nemotron-nano")
    assert model_client._resolve_model("nano") == "nvidia/nemotron-nano"
