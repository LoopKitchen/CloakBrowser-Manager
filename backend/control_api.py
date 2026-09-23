"""Control lease and per-run VNC token endpoints.

The lease is the farm's input mutex: one profile is IDLE, AGENT, or HUMAN
controlled. A human preempts an agent; an agent never preempts a human.
The VNC token lets a browser open ``/api/profiles/{id}/vnc`` without the
Manager's master bearer — it is minted only against an active HUMAN lease and
stops working the moment that lease is released or expires.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from .control_lease import (
    ControlLeaseConflictError,
    ControlLeaseManager,
    ControlLeaseMismatchError,
)

router = APIRouter()
control_leases = ControlLeaseManager()

VNC_TOKEN_TTL_S = 120


class AcquireControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holder: Literal["agent", "human"]
    ttl_s: float = Field(gt=0, allow_inf_nan=False)


class ReleaseControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_id: str = Field(min_length=1)


class VncTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_id: str = Field(min_length=1)
    run_id: str = ""
    user: str = ""


def _lease_wire(profile_id: str) -> dict[str, object]:
    lease = control_leases.status(profile_id)
    if lease is None:
        return {"lease_id": None, "state": "idle", "holder": None, "expires_at": None}
    return lease.to_wire()


@router.post("/api/profiles/{profile_id}/control/acquire")
async def acquire_control(profile_id: str, request: AcquireControlRequest):
    try:
        lease = control_leases.acquire(profile_id, request.holder, request.ttl_s)
    except ControlLeaseConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return lease.to_wire()


@router.post("/api/profiles/{profile_id}/control/release", status_code=204)
async def release_control(profile_id: str, request: ReleaseControlRequest):
    try:
        control_leases.release(profile_id, request.lease_id)
    except ControlLeaseMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


@router.get("/api/profiles/{profile_id}/control/status")
async def control_status(profile_id: str):
    return _lease_wire(profile_id)


# ── Per-run VNC tokens ────────────────────────────────────────────────────────


def _vnc_key(secret: str) -> bytes:
    return secret.encode()


def _vnc_sign(key: bytes, payload: str) -> str:
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()[:24]


def mint_vnc_token(
    secret: str,
    profile_id: str,
    lease_id: str,
    run_id: str,
    user: str,
    *,
    now: float | None = None,
) -> tuple[str, float]:
    ts = int(now if now is not None else time.time())
    payload = f"{ts}.{profile_id}.{lease_id}.{run_id}.{user}"
    return f"{payload}.{_vnc_sign(_vnc_key(secret), payload)}", ts + VNC_TOKEN_TTL_S


def verify_vnc_token(
    secret: str,
    token: str,
    profile_id: str,
    *,
    now: float | None = None,
) -> str | None:
    """Return the bound lease_id when the token is authentic and unexpired."""
    parts = token.split(".")
    if len(parts) < 5:
        return None
    ts_s, tok_profile, lease_id = parts[0], parts[1], parts[2]
    run_id, user, sig = ".".join(parts[3:-2]), parts[-2], parts[-1]
    payload = f"{ts_s}.{tok_profile}.{lease_id}.{run_id}.{user}"
    if tok_profile != profile_id:
        return None
    if not hmac.compare_digest(sig, _vnc_sign(_vnc_key(secret), payload)):
        return None
    try:
        ts = int(ts_s)
    except ValueError:
        return None
    if (now if now is not None else time.time()) > ts + VNC_TOKEN_TTL_S:
        return None
    return lease_id


@router.post("/api/profiles/{profile_id}/vnc/token")
async def create_vnc_token(profile_id: str, request: VncTokenRequest):
    from . import main as _main

    if not _main.AUTH_TOKEN:
        raise HTTPException(status_code=503, detail="auth disabled; VNC token meaningless")
    lease = control_leases.status(profile_id)
    if lease is None or lease.holder != "human" or lease.lease_id != request.lease_id:
        raise HTTPException(status_code=409, detail="no active human lease with that id")
    token, expires_at = mint_vnc_token(
        _main.AUTH_TOKEN, profile_id, request.lease_id, request.run_id, request.user
    )
    return {"vnc_token": token, "expires_at": expires_at}
