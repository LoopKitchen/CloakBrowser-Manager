"""Exclusive per-profile control leases for agent and human input."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

ControlHolder = Literal["agent", "human"]
ControlState = Literal["idle", "agent", "human"]


class ControlLeaseError(RuntimeError):
    """Base class for control-lease transition failures."""


class ControlLeaseConflictError(ControlLeaseError):
    """Raised when an agent tries to take control from a human."""


class ControlLeaseMismatchError(ControlLeaseError):
    """Raised when release does not present the active lease identifier."""


@dataclass(frozen=True, slots=True)
class ControlLease:
    lease_id: str
    state: Literal["agent", "human"]
    holder: ControlHolder
    expires_at: float

    def to_wire(self) -> dict[str, str]:
        return {
            "lease_id": self.lease_id,
            "state": self.state,
            "holder": self.holder,
            "expires_at": datetime.fromtimestamp(self.expires_at, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }


class ControlLeaseManager:
    """Atomic in-memory IDLE/AGENT/HUMAN FSM keyed by profile id.

    Human acquisition preempts an agent lease. An agent cannot preempt a human;
    the human must release its exact lease id before the agent reacquires.
    Expired leases are discarded at every lease boundary.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        lease_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._clock = clock
        self._lease_id_factory = lease_id_factory or (lambda: str(uuid.uuid4()))
        self._leases: dict[str, ControlLease] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        profile_id: str,
        holder: ControlHolder,
        ttl_s: float,
    ) -> ControlLease:
        if holder not in ("agent", "human"):
            raise ValueError("holder must be 'agent' or 'human'")
        if not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("ttl_s must be a positive finite number")

        with self._lock:
            now = self._clock()
            current = self._active_lease(profile_id, now)
            if current and current.holder == "human" and holder == "agent":
                raise ControlLeaseConflictError(
                    f"Profile {profile_id} is controlled by a human"
                )

            lease = ControlLease(
                lease_id=self._lease_id_factory(),
                state=holder,
                holder=holder,
                expires_at=now + ttl_s,
            )
            self._leases[profile_id] = lease
            return lease

    def release(self, profile_id: str, lease_id: str) -> None:
        with self._lock:
            current = self._active_lease(profile_id, self._clock())
            if current is None or current.lease_id != lease_id:
                raise ControlLeaseMismatchError(
                    f"Lease {lease_id} is not active for profile {profile_id}"
                )
            del self._leases[profile_id]

    def status(self, profile_id: str) -> ControlLease | None:
        with self._lock:
            return self._active_lease(profile_id, self._clock())

    def state(self, profile_id: str) -> ControlState:
        lease = self.status(profile_id)
        return lease.state if lease else "idle"

    def agent_can_dispatch(self, profile_id: str, lease_id: str) -> bool:
        """Return whether this exact agent lease may dispatch CDP input."""
        lease = self.status(profile_id)
        return bool(lease and lease.holder == "agent" and lease.lease_id == lease_id)

    def release_profile(self, profile_id: str) -> None:
        """Release any lease because its browser profile stopped."""
        with self._lock:
            self._leases.pop(profile_id, None)

    def clear(self) -> None:
        with self._lock:
            self._leases.clear()

    def _active_lease(self, profile_id: str, now: float) -> ControlLease | None:
        lease = self._leases.get(profile_id)
        if lease and lease.expires_at <= now:
            del self._leases[profile_id]
            return None
        return lease
