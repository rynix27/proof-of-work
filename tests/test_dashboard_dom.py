"""Runs the real dashboard in a DOM (jsdom) against hostile API data.
Skipped automatically if Node.js or jsdom is not installed:
    cd tests/browser && npm install
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent / "browser"


def _jsdom_path():
    for cand in (os.environ.get("JSDOM_PATH"), str(HERE / "node_modules" / "jsdom")):
        if cand and Path(cand).exists():
            return cand
    return None


@pytest.mark.skipif(shutil.which("node") is None or _jsdom_path() is None,
                    reason="needs node + jsdom (cd tests/browser && npm install)")
def test_dashboard_renders_hostile_data_as_inert_text():
    r = subprocess.run(["node", str(HERE / "xss_harness.js")], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "JSDOM_PATH": _jsdom_path()})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS no payload executed" in r.stdout and "FAIL" not in r.stdout
