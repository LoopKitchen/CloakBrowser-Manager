"""Tests for per-profile agent/human control leases."""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from backend import main
from backend.control_lease import (
    ControlLeaseConflictError,
    ControlLeaseManager,
    ControlLeaseMismatchError,
)
from backend.teach_replay_api import control_leases


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _clear_shared_leases():
    control_leases.clear()
    yield
    control_leases.clear()


def _manager(clock: _Clock) -> ControlLeaseManager:
    ids = iter(("lease-1", "lease-2", "lease-3", "lease-4"))
    return ControlLeaseManager(clock=clock, lease_id_factory=lambda: next(ids))


def test_idle_profile_can_be_acquired_by_either_holder() -> None:
    clock = _Clock()
    leases = _manager(clock)

    agent = leases.acquire("agent-profile", "agent", 30)
    human = leases.acquire("human-profile", "human", 45)

    assert agent.to_wire() == {
        "lease_id": "lease-1",
        "state": "agent",
        "holder": "agent",
        "expires_at": "1970-01-01T00:17:10Z",
    }
    assert human.state == "human"
    assert leases.state("missing-profile") == "idle"


def test_human_takeover_preempts_agent_and_invalidates_agent_lease() -> None:
    clock = _Clock()
    leases = _manager(clock)
    agent = leases.acquire("profile-1", "agent", 60)

    human = leases.acquire("profile-1", "human", 120)

    assert human.lease_id != agent.lease_id
    assert leases.status("profile-1") == human
    assert not leases.agent_can_dispatch("profile-1", agent.lease_id)
    with pytest.raises(ControlLeaseMismatchError):
        leases.release("profile-1", agent.lease_id)
    assert leases.status("profile-1") == human


def test_agent_reacquires_only_after_human_handback() -> None:
    clock = _Clock()
    leases = _manager(clock)
    human = leases.acquire("profile-1", "human", 60)

    with pytest.raises(ControlLeaseConflictError):
        leases.acquire("profile-1", "agent", 60)

    leases.release("profile-1", human.lease_id)
    agent = leases.acquire("profile-1", "agent", 60)
    assert leases.agent_can_dispatch("profile-1", agent.lease_id)


def test_lease_ttl_expires_back_to_idle() -> None:
    clock = _Clock()
    leases = _manager(clock)
    lease = leases.acquire("profile-1", "agent", 10)
    clock.now = lease.expires_at

    assert leases.status("profile-1") is None
    assert leases.state("profile-1") == "idle"
    with pytest.raises(ControlLeaseMismatchError):
        leases.release("profile-1", lease.lease_id)


def test_release_requires_exact_active_lease_id() -> None:
    clock = _Clock()
    leases = _manager(clock)
    lease = leases.acquire("profile-1", "agent", 60)

    with pytest.raises(ControlLeaseMismatchError):
        leases.release("profile-1", "wrong-id")
    assert leases.status("profile-1") == lease

    leases.release("profile-1", lease.lease_id)
    assert leases.state("profile-1") == "idle"


def test_release_profile_clears_any_holder() -> None:
    clock = _Clock()
    leases = _manager(clock)
    leases.acquire("profile-1", "human", 60)

    leases.release_profile("profile-1")

    assert leases.state("profile-1") == "idle"


def test_control_api_matches_wire_contract_and_reports_conflict(
    app_client: TestClient,
) -> None:
    idle = app_client.get("/api/profiles/profile-1/control/status")
    assert idle.status_code == 200
    assert idle.json() == {
        "lease": {
            "lease_id": None,
            "state": "idle",
            "holder": None,
            "expires_at": None,
        }
    }

    acquired = app_client.post(
        "/api/profiles/profile-1/control/acquire",
        json={"holder": "human", "ttl_s": 60},
    )
    assert acquired.status_code == 200
    lease = acquired.json()["lease"]
    assert set(lease) == {"lease_id", "state", "holder", "expires_at"}
    assert lease["state"] == "human"
    assert lease["holder"] == "human"

    conflict = app_client.post(
        "/api/profiles/profile-1/control/acquire",
        json={"holder": "agent", "ttl_s": 60},
    )
    assert conflict.status_code == 409

    wrong_release = app_client.post(
        "/api/profiles/profile-1/control/release",
        json={"lease_id": "not-the-active-lease"},
    )
    assert wrong_release.status_code == 409

    released = app_client.post(
        "/api/profiles/profile-1/control/release",
        json={"lease_id": lease["lease_id"]},
    )
    assert released.status_code == 204
    assert released.content == b""
    assert (
        app_client.get("/api/profiles/profile-1/control/status").json()["lease"][
            "state"
        ]
        == "idle"
    )


def test_profile_stop_releases_control_lease(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = app_client.post("/api/profiles", json={"name": "Lease Stop"})
    profile_id = created.json()["id"]
    main.browser_mgr.running[profile_id] = MagicMock()
    monkeypatch.setattr(main.browser_mgr, "stop", AsyncMock())
    control_leases.acquire(profile_id, "human", 60)

    try:
        response = app_client.post(f"/api/profiles/{profile_id}/stop")
        assert response.status_code == 200
        assert control_leases.state(profile_id) == "idle"
    finally:
        main.browser_mgr.running.pop(profile_id, None)


def test_rfb_gate_drops_input_while_preserving_view_requests() -> None:
    key = struct.pack(">BBxxI", 4, 1, 0x61)
    pointer = struct.pack(">BBHH", 5, 1, 100, 200)
    clipboard_text = b"hello"
    clipboard = struct.pack(">BxxxI", 6, len(clipboard_text)) + clipboard_text
    framebuffer_request = struct.pack(">BBHHHH", 3, 1, 0, 0, 1920, 1080)

    profile_id = "profile-1"
    idle = main._filter_rfb_client_messages(
        key + framebuffer_request + pointer + clipboard,
        input_allowed=control_leases.state(profile_id) == "human",
    )
    control_leases.acquire(profile_id, "agent", 60)
    blocked = main._filter_rfb_client_messages(
        key + framebuffer_request + pointer + clipboard,
        input_allowed=control_leases.state(profile_id) == "human",
    )
    control_leases.release_profile(profile_id)
    control_leases.acquire(profile_id, "human", 60)
    allowed = main._filter_rfb_client_messages(
        key + framebuffer_request + pointer + clipboard,
        input_allowed=control_leases.state(profile_id) == "human",
    )

    assert idle == framebuffer_request
    assert blocked == framebuffer_request
    assert allowed[0] == 4
    assert allowed[8:18] == framebuffer_request
    assert allowed[18] == 5
    assert allowed[29] == 6
