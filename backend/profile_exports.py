"""Bounded snapshots of stopped browser state; archives contain session credentials."""

from __future__ import annotations

import asyncio
import json
import ipaddress
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
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.to_thread(shutil.rmtree, Path(self.path).parent, True)


def _build(root: Path, manifest: dict, skip: set[str], limit: int) -> Path:
    if not root.is_dir() or root.is_symlink():
        raise HTTPException(409, "Profile has no exportable browser state")
    directory = Path(tempfile.mkdtemp(prefix="cloak-export-"))
    archive = directory / "profile.zip"
    try:
        total = 0
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
            out.writestr("profile.json", json.dumps(manifest, indent=2))
            for parent, dirs, files in os.walk(root, followlinks=False):
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
    limit = int(os.environ.get("CLOAK_PROFILE_EXPORT_MAX_BYTES", 512 * 1024 * 1024))
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
    async with _SLOTS:
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
                await asyncio.to_thread(shutil.rmtree, task.result().parent, True)
            raise asyncio.CancelledError
        return ArchiveResponse(
            task.result(),
            media_type="application/zip",
            filename="cloak-profile.zip",
            headers={"Cache-Control": "no-store"},
        )
