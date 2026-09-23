"""Tests for per-profile agent/human control leases."""

from __future__ import annotations

import pytest

from backend.control_api import control_leases
from backend.control_lease import (
    ControlLeaseConflictError,
    ControlLeaseManager,
    ControlLeaseMismatchError,
)


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

    human = leases.acquire("profile-1", "human", 60)

    assert human.holder == "human"
    assert leases.state("profile-1") == "human"
    with pytest.raises(ControlLeaseMismatchError):
        leases.release("profile-1", agent.lease_id)


def test_agent_cannot_preempt_human() -> None:
    clock = _Clock()
    leases = _manager(clock)
    leases.acquire("profile-1", "human", 60)

    with pytest.raises(ControlLeaseConflictError):
        leases.acquire("profile-1", "agent", 60)


def test_expired_lease_returns_to_idle() -> None:
    clock = _Clock()
    leases = _manager(clock)
    lease = leases.acquire("profile-1", "human", 30)

    clock.now += 31

    assert leases.state("profile-1") == "idle"
    with pytest.raises(ControlLeaseMismatchError):
        leases.release("profile-1", lease.lease_id)
    # After expiry the agent may acquire again.
    leases.acquire("profile-1", "agent", 30)
    assert leases.state("profile-1") == "agent"


def test_release_requires_exact_lease_id() -> None:
    clock = _Clock()
    leases = _manager(clock)
    lease = leases.acquire("profile-1", "agent", 30)

    with pytest.raises(ControlLeaseMismatchError):
        leases.release("profile-1", "other-lease")
    leases.release("profile-1", lease.lease_id)

    assert leases.state("profile-1") == "idle"


def test_agent_can_dispatch_only_with_own_lease() -> None:
    clock = _Clock()
    leases = _manager(clock)
    lease = leases.acquire("profile-1", "agent", 30)

    assert leases.agent_can_dispatch("profile-1", lease.lease_id)
    assert not leases.agent_can_dispatch("profile-1", "other")

    leases.acquire("profile-1", "human", 30)
    assert not leases.agent_can_dispatch("profile-1", lease.lease_id)


def test_release_profile_drops_any_lease() -> None:
    clock = _Clock()
    leases = _manager(clock)
    leases.acquire("profile-1", "agent", 30)

    leases.release_profile("profile-1")

    assert leases.state("profile-1") == "idle"


def test_invalid_holder_and_ttl_rejected() -> None:
    clock = _Clock()
    leases = _manager(clock)

    with pytest.raises(ValueError):
        leases.acquire("profile-1", "robot", 30)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        leases.acquire("profile-1", "agent", 0)
    with pytest.raises(ValueError):
        leases.acquire("profile-1", "agent", float("inf"))
