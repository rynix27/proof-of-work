"""Regression tests for defects found in the second audit pass (CSRF, token guessing,
stale-run requests, body limits, read-rate limiting, concurrency of control calls)."""
import io
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

import server
from core import config, events
from core.control import control


@pytest.fixture
def client(isolated_data_dir):
    events.reset_log()
    return server.app.test_client()


EVIL = {"Origin": "https://evil.example", "Content-Type": "text/plain"}


@pytest.mark.parametrize("path", ["/control/pause", "/control/resume", "/control/speed", "/control/reset",
                                  "/control/save_replay", "/inject_task"])
def test_cross_origin_posts_are_refused_in_every_mode(client, monkeypatch, path):
    for mode in ("open", "token", "public"):
        monkeypatch.setattr(config, "CONTROL_TOKEN", "tok" if mode == "token" else "")
        monkeypatch.setattr(config, "PUBLIC_MODE", mode == "public")
        r = client.post(path, data="{}", headers=EVIL)
        assert r.status_code == 403, (path, mode, r.status_code)
    assert not control.is_paused()


def test_csrf_cannot_pause_or_reset_an_open_local_server(client):
    before = events.current_run_id()
    assert client.post("/control/pause", headers=EVIL).status_code == 403
    assert client.post("/control/reset", headers=EVIL).status_code == 403
    assert not control.is_paused() and events.current_run_id() == before


def test_null_origin_and_cross_site_fetch_metadata_refused(client):
    assert client.post("/control/pause", headers={"Origin": "null"}).status_code == 403
    assert client.post("/control/pause", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_same_origin_and_non_browser_clients_still_work(client):
    assert client.post("/control/pause", headers={"Origin": "http://localhost"}).status_code == 200
    assert client.post("/control/resume").status_code == 200                       # curl: no Origin
    assert client.post("/control/pause", headers={"Origin": "http://localhost",
                                                  "Sec-Fetch-Site": "same-origin"}).status_code == 200


def test_allowed_origins_allowlist(client, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["https://demo.example.com"])
    assert client.post("/control/pause", headers={"Origin": "https://demo.example.com/"}).status_code == 200
    assert client.post("/control/pause", headers={"Origin": "https://demo.example.com.evil.io"}).status_code == 403


def test_wrong_token_attempts_are_throttled_and_do_not_reveal_correctness(client, monkeypatch):
    monkeypatch.setattr(config, "CONTROL_TOKEN", "the-real-token")
    monkeypatch.setattr(config, "AUTH_FAIL_MAX_PER_MINUTE", 5)
    codes = [client.post("/control/pause", headers={"X-Control-Token": f"guess{i}"}).status_code for i in range(8)]
    assert codes == [401] * 5 + [429] * 3
    # while throttled even the correct token is refused (a guesser cannot tell when they hit it)
    r = client.post("/control/pause", headers={"X-Control-Token": "the-real-token"})
    assert r.status_code == 429 and r.headers["Retry-After"].isdigit()
    other = client.post("/control/pause", headers={"X-Control-Token": "the-real-token"},
                        environ_overrides={"REMOTE_ADDR": "8.8.8.8"})
    assert other.status_code == 200                                   # a different client is unaffected


def test_successful_auth_does_not_consume_failure_budget(client, monkeypatch):
    monkeypatch.setattr(config, "CONTROL_TOKEN", "tok")
    monkeypatch.setattr(config, "AUTH_FAIL_MAX_PER_MINUTE", 2)
    for _ in range(10):
        assert client.post("/control/resume", headers={"X-Control-Token": "tok"}).status_code == 200


def test_stale_run_requests_are_refused_with_409(client):
    old = events.current_run_id()
    events.reset_log()                                       # a new run starts
    new = events.current_run_id()
    for body_or_header in ({"headers": {"X-Run-Id": old}}, {"json": {"run_id": old}}):
        r = client.post("/control/pause", **body_or_header)
        assert r.status_code == 409 and r.json["current_run_id"] == new
    assert not control.is_paused()
    assert client.post("/control/pause", headers={"X-Run-Id": new}).status_code == 200
    control.resume()
    assert client.post("/control/pause").status_code == 200   # run id is optional (curl users)
    control.resume()


def test_reset_is_exempt_from_the_stale_guard(client):
    r = client.post("/control/reset", headers={"X-Run-Id": "ancient"}, json={"config": {"max_rounds": 1}})
    assert r.status_code == 200
    control.round_delay = 0


def test_run_id_of_wrong_type_is_ignored_not_a_crash(client):
    assert client.post("/control/pause", json={"run_id": 123}).status_code == 200
    assert client.post("/control/resume", json=[1, 2]).status_code == 200     # resume takes no body: ignored


def test_read_endpoints_are_rate_limited_per_client(client, monkeypatch):
    monkeypatch.setattr(config, "READ_MAX_PER_MINUTE", 10)
    codes = [client.get("/state").status_code for _ in range(13)]
    assert codes == [200] * 10 + [429] * 3
    assert client.get("/healthz").status_code == 200                 # health checks are never throttled
    assert client.get("/state", environ_overrides={"REMOTE_ADDR": "7.7.7.7"}).status_code == 200


def test_flask_enforces_max_content_length(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_REQUEST_BYTES", 300)
    client.get("/healthz")
    assert server.app.config["MAX_CONTENT_LENGTH"] == 300
    body = b'{"prompt": "' + b"x" * 400 + b'"}'
    r = client.post("/inject_task", input_stream=io.BytesIO(body), content_type="application/json")
    assert r.status_code == 413 and r.json["error"] == "request too large"


def test_concurrent_control_requests_leave_consistent_state(client):
    errors, results = [], []

    def hammer(i):
        try:
            for j in range(40):
                path = ("/control/pause", "/control/resume", "/control/speed")[(i + j) % 3]
                body = {"seconds": 1 + (j % 5)} if path.endswith("speed") else None
                results.append(client.post(path, json=body).status_code)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and set(results) == {200}
    assert isinstance(control.is_paused(), bool) and 0.1 <= control.round_delay <= 30
    control.resume()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_waitress_rejects_oversized_bodies_before_the_app_sees_them(isolated_data_dir):
    """Real HTTP server, real socket: the production server's own limit is in force."""
    from waitress.server import create_server
    port = _free_port()
    srv = create_server(server.app, host="127.0.0.1", port=port, max_request_body_size=1024)

    def serve():
        try:
            srv.run()
        except OSError:        # closing the listening socket during teardown ends the loop
            pass
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    time.sleep(0.3)
    try:
        big = json.dumps({"prompt": "x" * 5000}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/inject_task", data=big,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code == 413
        small = urllib.request.Request(f"http://127.0.0.1:{port}/inject_task", data=b'{"prompt": "hi"}',
                                       headers={"Content-Type": "application/json"}, method="POST")
        assert urllib.request.urlopen(small, timeout=5).status == 200
    finally:
        srv.close()
        t.join(timeout=2)
