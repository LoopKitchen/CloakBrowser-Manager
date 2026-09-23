"""Tests for the /control/* endpoints, per-run VNC tokens and RFB input gating."""

from __future__ import annotations

import struct

import pytest
from starlette.testclient import TestClient

from backend import main
from backend.control_api import control_leases, mint_vnc_token, verify_vnc_token


@pytest.fixture(autouse=True)
def _clear_shared_leases():
    control_leases.clear()
    yield
    control_leases.clear()


# ── Lease endpoints ───────────────────────────────────────────────────────────


def test_acquire_release_status_roundtrip(app_client: TestClient):
    resp = app_client.post(
        "/api/profiles/p1/control/acquire", json={"holder": "agent", "ttl_s": 60}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "agent"
    assert body["holder"] == "agent"
    assert body["lease_id"]

    status = app_client.get("/api/profiles/p1/control/status").json()
    assert status["lease_id"] == body["lease_id"]
    assert status["state"] == "agent"

    resp = app_client.post(
        "/api/profiles/p1/control/release", json={"lease_id": body["lease_id"]}
    )
    assert resp.status_code == 204
    assert app_client.get("/api/profiles/p1/control/status").json()["state"] == "idle"


def test_agent_acquire_on_human_held_profile_conflicts(app_client: TestClient):
    app_client.post("/api/profiles/p1/control/acquire", json={"holder": "human", "ttl_s": 60})
    resp = app_client.post(
        "/api/profiles/p1/control/acquire", json={"holder": "agent", "ttl_s": 60}
    )
    assert resp.status_code == 409


def test_release_with_wrong_lease_id_conflicts(app_client: TestClient):
    app_client.post("/api/profiles/p1/control/acquire", json={"holder": "agent", "ttl_s": 60})
    resp = app_client.post(
        "/api/profiles/p1/control/release", json={"lease_id": "bogus"}
    )
    assert resp.status_code == 409


def test_status_idle_shape(app_client: TestClient):
    assert app_client.get("/api/profiles/p1/control/status").json() == {
        "lease_id": None,
        "state": "idle",
        "holder": None,
        "expires_at": None,
    }


# ── Per-run VNC tokens ────────────────────────────────────────────────────────


def test_vnc_token_mint_requires_active_human_lease(app_client: TestClient, monkeypatch):
    monkeypatch.setattr(main, "AUTH_TOKEN", "master-secret")
    auth = {"Authorization": "Bearer master-secret"}

    resp = app_client.post(
        "/api/profiles/p1/vnc/token", json={"lease_id": "nope"}, headers=auth
    )
    assert resp.status_code == 409

    control_leases.acquire("p1", "agent", 60)
    agent = app_client.post(
        "/api/profiles/p1/vnc/token",
        json={"lease_id": control_leases.status("p1").lease_id},
        headers=auth,
    )
    assert agent.status_code == 409  # agent lease does not mint viewer tokens

    human = control_leases.acquire("p1", "human", 60)
    ok = app_client.post(
        "/api/profiles/p1/vnc/token",
        json={"lease_id": human.lease_id, "run_id": "r1", "user": "u1"},
        headers=auth,
    )
    assert ok.status_code == 200
    assert verify_vnc_token("master-secret", ok.json()["vnc_token"], "p1") == human.lease_id


def test_verify_vnc_token_rejects_tampering_and_expiry():
    token, _ = mint_vnc_token("s", "p1", "L1", "r1", "u1", now=1_000)

    assert verify_vnc_token("s", token, "p1", now=1_050) == "L1"
    assert verify_vnc_token("other", token, "p1", now=1_050) is None
    assert verify_vnc_token("s", token, "p2", now=1_050) is None
    assert verify_vnc_token("s", token + "x", "p1", now=1_050) is None
    assert verify_vnc_token("s", "garbage", "p1", now=1_050) is None
    # After the connect-window TTL the token is dead
    assert verify_vnc_token("s", token, "p1", now=1_000 + 121) is None


def test_check_vnc_ws_auth_accepts_only_live_human_lease(monkeypatch):
    monkeypatch.setattr(main, "AUTH_TOKEN", "master-secret")
    lease = control_leases.acquire("p1", "human", 60)
    token, _ = mint_vnc_token("master-secret", "p1", lease.lease_id, "r1", "u1")

    def scope(tok: str) -> dict:
        return {
            "type": "websocket",
            "path": "/api/profiles/p1/vnc",
            "query_string": f"vnc_token={tok}".encode(),
        }

    assert main._check_vnc_ws_auth(scope(token))
    assert not main._check_vnc_ws_auth(scope("bogus"))
    assert not main._check_vnc_ws_auth(
        {"type": "websocket", "path": "/api/profiles/p1/cdp", "query_string": b""}
    )

    # Token dies with the lease
    control_leases.release_profile("p1")
    assert not main._check_vnc_ws_auth(scope(token))


# ── RFB input gating ──────────────────────────────────────────────────────────


def _key_event() -> bytes:
    # type=4, down-flag + pad + keysym — 8 bytes total
    return struct.pack(">BBxxI", 4, 1, 0x41)


def _framebuffer_update_request() -> bytes:
    # type=3, incremental + geometry — 10 bytes
    return struct.pack(">BBHHHH", 3, 1, 0, 0, 100, 100)


def test_rfb_filter_drops_input_when_not_allowed():
    frame = _key_event() + _framebuffer_update_request()

    allowed = main._filter_rfb_client_messages(frame, input_allowed=True)
    blocked = main._filter_rfb_client_messages(frame, input_allowed=False)

    assert allowed == frame
    # KeyEvent (type 4) is dropped; FramebufferUpdateRequest (type 3) survives
    assert blocked == _framebuffer_update_request()


def test_rfb_filter_blocks_pointer_and_clipboard_when_not_allowed():
    pointer = struct.pack(">BBHH", 5, 1, 10, 20)
    cut_text = struct.pack(">BxxxI", 6, 3) + b"abc"
    frame = pointer + cut_text + _framebuffer_update_request()

    blocked = main._filter_rfb_client_messages(frame, input_allowed=False)

    assert _framebuffer_update_request() in blocked
    # 11-byte rewritten PointerEvent and ClientCutText are both gone
    assert len(blocked) == len(_framebuffer_update_request())
