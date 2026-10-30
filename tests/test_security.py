"""Static and behavioural security regression tests."""
import re
from pathlib import Path

import pytest

from core import config, events
from core.security import redact

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = [ROOT / "web" / "dashboard.js", ROOT / "web" / "dashboard.html"]

# APIs that parse a string as HTML or code. The frontend must never use them:
# model output, task prompts, agent names, epitaphs and replay filenames are untrusted.
FORBIDDEN = ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
             "setTimeout(\"", "setInterval(\"", "dangerouslySetInnerHTML", "srcdoc", "javascript:"]


@pytest.mark.parametrize("path", FRONTEND, ids=lambda p: p.name)
def test_frontend_never_uses_html_or_code_string_sinks(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    for token in FORBIDDEN:
        assert token not in text, f"{path.name} contains forbidden sink {token!r}"


def test_html_has_no_inline_script_or_inline_handlers_or_remote_script():
    html = (ROOT / "web" / "dashboard.html").read_text()
    scripts = re.findall(r"<script\b[^>]*>", html)
    assert scripts == ['<script src="/dashboard.js" defer>']
    assert not re.search(r"\son\w+\s*=", html), "inline event handler attribute found"
    assert "http://" not in html            # no insecure resources


def test_redact_removes_configured_key_and_bearer_tokens(monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "nb-KEY-abcdef123456")
    assert "abcdef123456" not in redact("auth failed for nb-KEY-abcdef123456 on host")
    assert "tok" not in redact("Authorization: Bearer tokentokentokentoken1234").replace("[REDACTED]", "")
    assert redact(None) == ""


def test_secrets_never_reach_the_event_log(isolated_data_dir, monkeypatch):
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "nb-KEY-abcdef123456")
    events.reset_log()
    events.emit("call_error", error="401 invalid key nb-KEY-abcdef123456")
    assert "abcdef123456" not in events.LOG_PATH.read_text()


def test_env_example_contains_no_real_looking_secret():
    text = (ROOT / ".env.example").read_text()
    for line in text.splitlines():
        if line.startswith("NEBIUS_API_KEY="):
            assert line.split("=", 1)[1].strip() in ("", "your-key-here", "your_key_here")


def test_gitignore_protects_env_and_data():
    gi = (ROOT / ".gitignore").read_text()
    assert ".env" in gi and "data/" in gi


def test_dockerignore_excludes_secrets_and_data():
    di = (ROOT / ".dockerignore").read_text()
    for item in (".env", "data", ".git"):
        assert item in di


def test_model_error_messages_are_redacted(monkeypatch):
    from tests._compat import httpx
    import openai
    from core import model_client
    monkeypatch.setattr(config, "NEBIUS_API_KEY", "nb-KEY-abcdef123456")
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    resp = httpx.Response(401, request=req)

    class C:
        class chat:
            class completions:
                @staticmethod
                def create(**k):
                    raise openai.AuthenticationError("bad key nb-KEY-abcdef123456", response=resp, body=None)
    with pytest.raises(model_client.ModelCallError) as e:
        model_client._call_with_retry(C, "m", [], "nano")
    assert "abcdef123456" not in str(e.value) and e.value.kind == "auth" and e.value.charged is False
