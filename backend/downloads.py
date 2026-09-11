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
from typing import Callable

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

    def __init__(self, profile_id: str, cancel: Callable[[str], None] | None = None) -> None:
        self.profile_id = profile_id
        # Called with a GUID to ask Chromium to stop a transfer that broke a limit.
        self._cancel = cancel
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
        db.create_artifact(
            artifact_id, self.profile_id, name, 0, kind="download", state="pending",
            picker_name=artifacts.reserve_picker_name(self.profile_id, name),
        )
        self._by_guid[guid] = artifact_id
        return artifact_id

    def progress(self, guid: str | None, state: str | None, received_bytes: int = 0) -> None:
        """Publish or discard a tracked download once the browser says it is done."""
        artifact_id = self._by_guid.get(guid or "")
        if artifact_id is None:
            return
        if state == "inProgress":
            # Refusing to record it would not stop Chromium, which would keep filling the
            # staging directory unseen. Stop the transfer instead.
            if received_bytes > artifacts.MAX_ARTIFACT_BYTES:
                self._fail(artifact_id, guid or "", "exceeded the maximum artifact size")
                self._by_guid.pop(guid or "", None)
                if self._cancel:
                    self._cancel(guid or "")
                artifacts.sync_picker_view(self.profile_id)
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
        # The size is only known for certain now, so the caps are checked against the real
        # bytes rather than against the zero we had when the transfer started.
        try:
            actual = source.stat().st_size
        except OSError as exc:
            logger.warning("Profile %s: download %r did not land: %s", self.profile_id, row["name"], exc)
            db.update_artifact(self.profile_id, artifact_id, state="failed", size=received_bytes)
            return
        if actual > artifacts.MAX_ARTIFACT_BYTES:
            self._fail(artifact_id, guid, "exceeded the maximum artifact size")
            return
        try:
            artifacts.check_quota(self.profile_id, actual)
        except artifacts.ArtifactQuotaExceeded as exc:
            self._fail(artifact_id, guid, str(exc))
            return
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, dest)
            size = dest.stat().st_size
        except OSError as exc:
            logger.warning("Profile %s: download %r did not land: %s", self.profile_id, row["name"], exc)
            db.update_artifact(self.profile_id, artifact_id, state="failed", size=received_bytes)
            return
        db.update_artifact(self.profile_id, artifact_id, state="ready", size=size)

    def _fail(self, artifact_id: str, guid: str, reason: str) -> None:
        """Record a refused or broken transfer and drop whatever it staged."""
        logger.warning("Profile %s: download refused — %s", self.profile_id, reason)
        (staging_dir(self.profile_id) / guid).unlink(missing_ok=True)
        db.update_artifact(self.profile_id, artifact_id, state="failed")

    def _discard(self, artifact_id: str, guid: str) -> None:
        (staging_dir(self.profile_id) / guid).unlink(missing_ok=True)
        artifacts.delete_file(self.profile_id, artifact_id)
        db.delete_artifact(self.profile_id, artifact_id)


def reconcile_interrupted(profile_id: str) -> int:
    """Fail any download left pending by a previous browser, and clear its staged bytes.

    A fresh browser will never resume them, so leaving them pending would show a transfer
    that is permanently in flight.
    """
    stranded = [row for row in db.list_artifacts(profile_id) if row["state"] == "pending"]
    for row in stranded:
        db.update_artifact(profile_id, row["id"], state="failed")
    staging = staging_dir(profile_id)
    if staging.is_dir():
        for leftover in staging.iterdir():
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
    return len(stranded)


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

    try:
        staging = staging_dir(profile_id)
        staging.mkdir(parents=True, exist_ok=True)
        stranded = reconcile_interrupted(profile_id)
        if stranded:
            logger.info("Profile %s: failed %d download(s) interrupted by a restart", profile_id, stranded)
        url = await _browser_websocket_url(cdp_port)
        async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
            def cancel(guid: str) -> None:
                asyncio.ensure_future(ws.send(json.dumps({
                    "id": 0, "method": "Browser.cancelDownload", "params": {"guid": guid},
                })))

            tracker = DownloadTracker(profile_id, cancel=cancel)
            await ws.send(json.dumps({
                "id": 1,
                "method": "Browser.setDownloadBehavior",
                "params": {
                    "behavior": "allowAndName",
                    "downloadPath": str(staging),
                    "eventsEnabled": True,
                },
            }))
            # Only claim capture is on once Chromium has actually accepted the configuration.
            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if ack.get("id") == 1 and "error" in ack:
                logger.warning(
                    "Profile %s: download capture unavailable: %s", profile_id, ack["error"]
                )
                return
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
