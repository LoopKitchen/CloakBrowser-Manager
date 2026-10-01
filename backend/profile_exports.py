"""Bounded snapshots of stopped browser state; archives contain session credentials."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

_SLOTS = asyncio.Semaphore(2)


class ArchiveResponse(FileResponse):
    def __init__(self, *args, slot: asyncio.Semaphore, **kwargs):
        super().__init__(*args, **kwargs)
        self._slot = slot

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await _discard_archive(Path(self.path))
            finally:
                self._slot.release()


async def _discard_archive(path: Path):
    cleanup = asyncio.create_task(asyncio.to_thread(shutil.rmtree, path.parent, True))
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    cleanup.result()
    if cancelled:
        raise asyncio.CancelledError


def _scan_failed(error: OSError):
    raise error


def _build(root: Path, manifest: dict, skip: set[str], limit: int) -> Path:
    if not root.is_dir() or root.is_symlink():
        raise HTTPException(409, "Profile has no exportable browser state")
    directory = Path(tempfile.mkdtemp(prefix="cloak-export-"))
    archive = directory / "profile.zip"
    try:
        total = 0
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
            out.writestr("profile.json", json.dumps(manifest, indent=2))
            for parent, dirs, files in os.walk(root, followlinks=False, onerror=_scan_failed):
                dirs[:] = [d for d in dirs if d not in skip and not (Path(parent) / d).is_symlink()]
                for name in files:
                    file = Path(parent) / name
                    info = file.lstat()
                    if name in skip or not stat.S_ISREG(info.st_mode):
                        continue
                    total += info.st_size
                    if total > limit:
                        raise HTTPException(413, "Browser state exceeds the export size limit")
                    out.write(file, "user-data/" + file.relative_to(root).as_posix())
        if archive.stat().st_size > limit:
            raise HTTPException(413, "Archive exceeds the export size limit")
        return archive
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


async def snapshot(profile: dict, manager, skip: set[str]) -> ArchiveResponse:
    try:
        limit = int(os.environ.get("CLOAK_PROFILE_EXPORT_MAX_BYTES", 512 * 1024 * 1024))
    except ValueError:
        raise HTTPException(503, "Profile export size limit must be an integer") from None
    if limit <= 0:
        raise HTTPException(503, "Profile exports are disabled")
    if manager.runtime.host_os != "linux" or manager.runtime.runtime_mode != "docker":
        raise HTTPException(409, "Portable exports require a Linux Docker profile")
    extra = profile.get("launch_args") or []
    for arg in extra:
        flag, _, value = arg.partition("=")
        safe = flag == "--fingerprint-platform" and value in {"windows", "macos", "linux"}
        if flag == "--fingerprint-webrtc-ip":
            try:
                ipaddress.ip_address(value)
                safe = True
            except ValueError:
                safe = value == "auto"
        if not safe:
            raise HTTPException(409, "Export does not support this profile's custom launch arguments")
    manifest = {
        "format": "cloak-profile-v1",
        "user_data_dir": "user-data",
        "chromium_version": manager.binary_version,
        "host_os": manager.runtime.host_os,
        "fingerprint_args": manager._build_fingerprint_args(profile) + extra,
        **{
            k: profile.get(k)
            for k in (
                "fingerprint_seed",
                "screen_width",
                "screen_height",
                "gpu_family",
                "geoip",
                "timezone",
                "locale",
                "color_scheme",
                "humanize",
                "human_preset",
                "allow_3p_cookies",
            )
        },
    }
    slots = _SLOTS
    await slots.acquire()
    try:
        task = asyncio.create_task(asyncio.to_thread(_build, Path(profile["user_data_dir"]), manifest, skip, limit))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                if not cancelled:
                    raise
        if cancelled:
            if not task.cancelled() and task.exception() is None:
                await _discard_archive(task.result())
            raise asyncio.CancelledError
        return ArchiveResponse(
            task.result(),
            slot=slots,
            media_type="application/zip",
            filename="cloak-profile.zip",
            headers={"Cache-Control": "no-store"},
        )
    except BaseException:
        slots.release()
        raise
