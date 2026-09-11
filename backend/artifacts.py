"""Per-profile file artifacts — files pushed into a profile, and files its browser downloaded.

Artifacts live OUTSIDE the profile's ``user_data_dir``: duplicating or exporting a profile's
browser state must not carry merchant documents with it, and an upload arriving mid-copy must
not mutate a directory that a snapshot is reading.
"""

from __future__ import annotations

import asyncio
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


# Most Linux filesystems cap a path component at 255 BYTES, not characters, so a legitimate
# non-ASCII name can be well under 255 characters and still be rejected.
_MAX_NAME_BYTES = 255


def _fit_bytes(stem: str, suffix: str, budget: int) -> str:
    """Trim ``stem`` so ``stem + suffix`` fits ``budget`` bytes, keeping the extension."""
    suffix_bytes = len(suffix.encode("utf-8"))
    room = max(1, budget - suffix_bytes)
    while len(stem.encode("utf-8")) > room:
        stem = stem[:-1]
    return f"{stem}{suffix}" if stem else suffix[-budget:]


def safe_filename(name: str | None, fallback: str = "file") -> str:
    """Sanitize an ORIGINAL filename kept as metadata; it never becomes a path component."""
    cleaned = unicodedata.normalize("NFC", (name or "").strip()).replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)
    if cleaned in ("", ".", ".."):
        return fallback
    if len(cleaned.encode("utf-8")) <= _MAX_NAME_BYTES:
        return cleaned
    stem, dot, ext = cleaned.rpartition(".")
    return _fit_bytes(stem or cleaned, f"{dot}{ext}" if dot else "", _MAX_NAME_BYTES)


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


def _flush_to_disk(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


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
                # Off the loop: slow container storage must not stall the rest of the Manager.
                await asyncio.to_thread(handle.write, chunk)
            await asyncio.to_thread(_flush_to_disk, handle)
        await asyncio.to_thread(os.replace, tmp, artifact_path(profile_id, artifact_id, name))
        return total
    except BaseException:
        tmp.unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        raise


# A page's "Choose File" opens the container's own GTK dialog, which starts in an empty $HOME
# and cannot reach an id-keyed path a human would ever type. These two helpers give that dialog
# a flat, real-named folder per profile and a sidebar shortcut to it. Cosmetic: never fatal.
PICKER_VIEW_DIRNAME = "files"


def _gtk_bookmarks_path() -> Path | None:
    """Only the Linux/Docker runtime owns the containerized GTK chooser we are bookmarking.
    Elsewhere ``$HOME`` is a real user's, and writing there would clobber their bookmarks."""
    if db.RUNTIME.runtime_mode != "docker":
        return None
    return Path.home() / ".config" / "gtk-3.0" / "bookmarks"


def picker_dir(profile_id: str) -> Path:
    return artifact_dir(profile_id) / PICKER_VIEW_DIRNAME


def reserve_picker_name(profile_id: str, name: str) -> str:
    """The name this artifact will keep in the file-chooser view, decided once.

    Deriving it at rebuild time would let an existing picker path start resolving to a
    DIFFERENT document once a sibling with the same name is added or removed.
    """
    taken = {row["picker_name"] or row["name"] for row in db.list_artifacts(profile_id)}
    stem, dot, ext = name.partition(".")
    candidate, counter = name, 1
    while candidate in taken:
        candidate = _fit_bytes(f"{stem} ({counter})", f"{dot}{ext}" if dot else "", _MAX_NAME_BYTES)
        counter += 1
    return candidate


def _refresh_gtk_bookmarks() -> None:
    """One sidebar entry per profile that has files, labelled with the profile name."""
    path = _gtk_bookmarks_path()
    if path is None:
        return
    lines = [
        f"file://{picker_dir(profile['id'])} Uploads \u2014 {profile['name']}"
        for profile in db.list_profiles()
        if picker_dir(profile["id"]).is_dir()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))


def sync_picker_view(profile_id: str) -> None:
    """Rebuild the profile's flat, real-named view and the file-chooser sidebar shortcut.

    Symlinks, so the bytes are stored once and the canonical id-keyed path stays the only
    place they live.
    """
    try:
        view = picker_dir(profile_id)
        view.mkdir(parents=True, exist_ok=True)
        wanted: dict[str, Path] = {}
        for row in db.list_artifacts(profile_id):
            target = artifact_path(profile_id, row["id"], row["name"])
            if target.is_file():
                wanted[row["picker_name"] or row["name"]] = target
        for existing in view.iterdir():
            if not existing.is_symlink():
                continue
            target = wanted.get(existing.name)
            if target is None or existing.resolve() != target.resolve():
                existing.unlink()
        for name, target in wanted.items():
            link = view / name
            if not link.is_symlink():
                link.symlink_to(target)
        _refresh_gtk_bookmarks()
    except OSError:
        # A convenience for the human-facing dialog; an upload must never fail over it.
        pass


def delete_file(profile_id: str, artifact_id: str) -> None:
    """Drop the artifact's whole directory — it holds exactly one file."""
    shutil.rmtree(artifact_item_dir(profile_id, artifact_id), ignore_errors=True)


def remove_profile_artifacts(profile_id: str) -> None:
    """Drop a profile's whole artifact directory; artifacts must not outlive their profile."""
    shutil.rmtree(artifact_dir(profile_id), ignore_errors=True)
