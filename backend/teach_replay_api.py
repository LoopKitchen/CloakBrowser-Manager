"""Shared API router for teach/replay manager capabilities."""

from __future__ import annotations

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


class AcquireControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holder: Literal["agent", "human"]
    ttl_s: float = Field(gt=0, allow_inf_nan=False)


class ReleaseControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_id: str = Field(min_length=1)


def _lease_payload(profile_id: str) -> dict[str, object]:
    lease = control_leases.status(profile_id)
    if lease:
        return {"lease": lease.to_wire()}
    return {
        "lease": {
            "lease_id": None,
            "state": "idle",
            "holder": None,
            "expires_at": None,
        }
    }


@router.post("/api/profiles/{profile_id}/control/acquire")
async def acquire_control(profile_id: str, request: AcquireControlRequest):
    try:
        lease = control_leases.acquire(profile_id, request.holder, request.ttl_s)
    except ControlLeaseConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"lease": lease.to_wire()}


@router.post("/api/profiles/{profile_id}/control/release")
async def release_control(profile_id: str, request: ReleaseControlRequest):
    try:
        control_leases.release(profile_id, request.lease_id)
    except ControlLeaseMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


@router.get("/api/profiles/{profile_id}/control/status")
async def control_status(profile_id: str):
    return _lease_payload(profile_id)
