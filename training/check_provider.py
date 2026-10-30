"""
Live provider smoke test -- RUN THIS FIRST when you get your Nebius key.

    USE_FAKE_MODEL=0 NEBIUS_API_KEY=... python -m training.check_provider

Checks (each prints PASS/FAIL, never prints your key):
  1. GET {NEBIUS_BASE_URL}models works with your key
  2. your configured NANO_MODEL / ULTRA_MODEL ids appear in that list
  3. a tiny chat completion returns NON-EMPTY content within MODEL_MAX_TOKENS
     (Nemotron 3 models are reasoning models; an exhausted token budget yields
     empty content -- raise MODEL_MAX_TOKENS if this fails)
  4. the reply parses as the JSON object the simulation expects
This is the only way the integration gets "verified": it has not been run by the
authors of this repository (no credentials in the build environment).
"""
import sys

from core import config
from core.security import redact
from core.validation import ValidationError, parse_json_object, validate_answer


def run(client=None) -> bool:
    ok_all = True

    def report(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    if client is None:
        if not config.NEBIUS_API_KEY:
            print("[FAIL] NEBIUS_API_KEY is not set")
            return False
        from openai import OpenAI
        client = OpenAI(api_key=config.NEBIUS_API_KEY, base_url=config.NEBIUS_BASE_URL, max_retries=0,
                        timeout=config.MODEL_TIMEOUT_SECONDS)
    print(f"base_url={config.NEBIUS_BASE_URL}")

    ids = set()
    try:
        ids = {m.id for m in client.models.list().data}
        report("list models", True, f"{len(ids)} models")
    except Exception as e:  # noqa: BLE001 -- diagnostic tool: report anything
        report("list models", False, redact(e))

    for label, model in (("nano", config.NANO_MODEL), ("ultra", config.ULTRA_MODEL)):
        if ids:
            report(f"{label} model id present ({model})", model in ids)

    try:
        resp = client.chat.completions.create(
            model=config.NANO_MODEL, max_tokens=config.MODEL_MAX_TOKENS, temperature=0,
            messages=[{"role": "system", "content": 'Respond ONLY with JSON: {"answer": "<answer>"}'},
                      {"role": "user", "content": "Question: What is 2 + 2?"}])
        content = resp.choices[0].message.content
        report("non-empty content", isinstance(content, str) and content.strip() != "",
               f"finish_reason={resp.choices[0].finish_reason}")
        try:
            validate_answer(parse_json_object(content))
            report("reply parses as expected JSON", True)
        except ValidationError as e:
            report("reply parses as expected JSON", False, e.code)
    except Exception as e:  # noqa: BLE001
        report("chat completion", False, redact(e))
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
