"""Per-profile file artifacts — files pushed into a profile, and files its browser downloaded.

Artifacts live OUTSIDE the profile's ``user_data_dir``: duplicating or exporting a profile's
browser state must not carry merchant documents with it, and an upload arriving mid-copy must
not mutate a directory that a snapshot is reading.
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Awaitable, Callable

from . import database as db

# Caps are enforced as bytes arrive. Content-Length is client-supplied and proves nothing.
MAX_ARTIFACT_BYTES = int(os.environ.get("ARTIFACT_MAX_BYTES", 256 * 1024 * 1024))
MAX_ARTIFACTS_PER_PROFILE = int(os.environ.get("ARTIFACT_MAX_COUNT", 200))
MAX_PROFILE_ARTIFACT_BYTES = int(os.environ.get("ARTIFACT_MAX_TOTAL_BYTES", 2 * 1024 * 1024 * 1024))

_CHUNK = 1024 * 1024
_INCOMING_PREFIX = ".incoming-"


class ArtifactTooLarge(Exception):
    """A single artifact exceeded MAX_ARTIFACT_BYTES."""


class ArtifactQuotaExceeded(Exception):
    """The profile's artifact count or total size cap would be exceeded."""


def artifacts_root() -> Path:
    return Path(db.DATA_DIR) / "artifacts"


def artifact_dir(profile_id: str) -> Path:
    return artifacts_root() / profile_id


def artifact_item_dir(profile_id: str, artifact_id: str) -> Path:
    """Each artifact owns a directory named by its id — that id, not the filename, is the
    trust boundary, so a crafted name cannot escape it."""
    return artifact_dir(profile_id) / artifact_id


def artifact_path(profile_id: str, artifact_id: str, name: str) -> Path:
    """The stored file keeps its real (sanitized) name inside the artifact's own directory,
    so a page receiving it by CDP sees the filename and extension a portal expects."""
    return artifact_item_dir(profile_id, artifact_id) / safe_filename(name)


def safe_filename(name: str | None, fallback: str = "file") -> str:
    """Sanitize an ORIGINAL filename kept as metadata; it never becomes a path component."""
    cleaned = unicodedata.normalize("NFC", (name or "").strip()).replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)
    return fallback if cleaned in ("", ".", "..") else cleaned[:255]


def check_quota(profile_id: str, incoming: int = 0) -> None:
    """Raise if another artifact of ``incoming`` bytes would breach the profile's caps."""
    rows = db.list_artifacts(profile_id)
    if len(rows) >= MAX_ARTIFACTS_PER_PROFILE:
        raise ArtifactQuotaExceeded(
            f"profile already holds {len(rows)} artifacts (max {MAX_ARTIFACTS_PER_PROFILE})"
        )
    used = sum(int(row["size"] or 0) for row in rows)
    if used + incoming > MAX_PROFILE_ARTIFACT_BYTES:
        raise ArtifactQuotaExceeded(
            f"profile artifact storage would reach {used + incoming} bytes "
            f"(max {MAX_PROFILE_ARTIFACT_BYTES})"
        )


async def save_stream(
    profile_id: str, artifact_id: str, name: str, read: Callable[[int], Awaitable[bytes]]
) -> int:
    """Stream an upload to a temp file, then publish it atomically under ``artifact_id``.

    The cap is enforced while receiving, and any failure removes the partial file, so a
    half-written upload is never visible as an artifact.
    """
    directory = artifact_item_dir(profile_id, artifact_id)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f"{_INCOMING_PREFIX}{artifact_id}"
    total = 0
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            while True:
                chunk = await read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_ARTIFACT_BYTES:
                    raise ArtifactTooLarge(
                        f"upload exceeds {MAX_ARTIFACT_BYTES} bytes"
                    )
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, artifact_path(profile_id, artifact_id, name))
        return total
    except BaseException:
        tmp.unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        raise


def delete_file(profile_id: str, artifact_id: str) -> None:
    """Drop the artifact's whole directory — it holds exactly one file."""
    shutil.rmtree(artifact_item_dir(profile_id, artifact_id), ignore_errors=True)


def remove_profile_artifacts(profile_id: str) -> None:
    """Drop a profile's whole artifact directory; artifacts must not outlive their profile."""
    shutil.rmtree(artifact_dir(profile_id), ignore_errors=True)
