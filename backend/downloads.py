"""Capture what a profile's browser downloads, as artifacts.

Chromium is told to name downloads by GUID (``allowAndName``) and to emit progress events, so
an artifact is published only once the browser reports the transfer finished. A filename
appearing in a directory proves nothing while a download is still running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import httpx

from . import artifacts
from . import database as db

logger = logging.getLogger("cloakbrowser.manager.downloads")

STAGING_DIRNAME = ".downloads"


def staging_dir(profile_id: str) -> Path:
    """Where Chromium drops GUID-named files before they are published."""
    return artifacts.artifact_dir(profile_id) / STAGING_DIRNAME


class DownloadTracker:
    """Turns one profile's CDP download events into artifact rows.

    Deliberately free of websocket plumbing so the state machine is testable without a browser.
    """

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        self._by_guid: dict[str, str] = {}

    def begin(self, guid: str | None, suggested_filename: str | None) -> str | None:
        """Record a started download as a pending artifact. Returns its id, or None if refused."""
        if not guid:
            return None
        name = artifacts.safe_filename(suggested_filename, fallback="download")
        try:
            artifacts.check_quota(self.profile_id)
        except artifacts.ArtifactQuotaExceeded as exc:
            logger.warning("Profile %s: not capturing %r: %s", self.profile_id, name, exc)
            return None
        artifact_id = db.new_artifact_id()
        db.create_artifact(artifact_id, self.profile_id, name, 0, kind="download", state="pending")
        self._by_guid[guid] = artifact_id
        return artifact_id

    def progress(self, guid: str | None, state: str | None, received_bytes: int = 0) -> None:
        """Publish or discard a tracked download once the browser says it is done."""
        artifact_id = self._by_guid.get(guid or "")
        if artifact_id is None or state == "inProgress":
            return
        self._by_guid.pop(guid or "", None)
        if state == "completed":
            self._publish(artifact_id, guid or "", received_bytes)
        else:
            self._discard(artifact_id, guid or "")
        artifacts.sync_picker_view(self.profile_id)

    def _publish(self, artifact_id: str, guid: str, received_bytes: int) -> None:
        row = db.get_artifact(self.profile_id, artifact_id)
        if row is None:
            return
        source = staging_dir(self.profile_id) / guid
        dest = artifacts.artifact_path(self.profile_id, artifact_id, row["name"])
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # `completed` does not promise the file is present; a missing one is a failure,
            # not an empty artifact.
            os.replace(source, dest)
            size = dest.stat().st_size
        except OSError as exc:
            logger.warning("Profile %s: download %r did not land: %s", self.profile_id, row["name"], exc)
            db.update_artifact(self.profile_id, artifact_id, state="failed", size=received_bytes)
            return
        db.update_artifact(self.profile_id, artifact_id, state="ready", size=size)

    def _discard(self, artifact_id: str, guid: str) -> None:
        (staging_dir(self.profile_id) / guid).unlink(missing_ok=True)
        artifacts.delete_file(self.profile_id, artifact_id)
        db.delete_artifact(self.profile_id, artifact_id)


async def _browser_websocket_url(cdp_port: int) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:{cdp_port}/json/version", timeout=5)
    return resp.json()["webSocketDebuggerUrl"]


def handle_event(tracker: DownloadTracker, message: dict) -> None:
    params = message.get("params") or {}
    method = message.get("method")
    if method == "Browser.downloadWillBegin":
        tracker.begin(params.get("guid"), params.get("suggestedFilename"))
    elif method == "Browser.downloadProgress":
        tracker.progress(params.get("guid"), params.get("state"), int(params.get("receivedBytes") or 0))


async def watch(profile_id: str, cdp_port: int) -> None:
    """Arm download capture on a running profile and publish what it downloads.

    Never raises into the launch path: losing capture must not cost the browser.
    """
    import websockets

    tracker = DownloadTracker(profile_id)
    try:
        staging = staging_dir(profile_id)
        staging.mkdir(parents=True, exist_ok=True)
        url = await _browser_websocket_url(cdp_port)
        async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
            await ws.send(json.dumps({
                "id": 1,
                "method": "Browser.setDownloadBehavior",
                "params": {
                    "behavior": "allowAndName",
                    "downloadPath": str(staging),
                    "eventsEnabled": True,
                },
            }))
            logger.info("Download capture armed for profile %s", profile_id)
            async for raw in ws:
                try:
                    handle_event(tracker, json.loads(raw))
                except Exception as exc:  # one bad event must not end capture
                    logger.warning("Profile %s: download event failed: %s", profile_id, exc)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Download capture stopped for profile %s: %s", profile_id, exc)
