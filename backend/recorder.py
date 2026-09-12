"""Per-profile recording of the manager's private X displays."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import signal
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from . import database as db
from .browser_manager import RunningProfile
from .teach_replay_api import router
from .vnc_manager import VNCManager

logger = logging.getLogger("cloakbrowser.manager.recorder")

MAX_RECORDING_SECONDS = 600
MIN_RECORDING_SECONDS = 0.5
RECORDING_FPS = 15


class RecordingError(RuntimeError):
    """Base class for recording lifecycle failures."""


class RecordingConflictError(RecordingError):
    """Raised when a profile already has a recording operation in progress."""


class RecordingNotActiveError(RecordingError):
    """Raised when a profile has no active recording to stop."""


class RecordingUnavailableError(RecordingError):
    """Raised when a running profile has no recordable X display."""


class RecordingProcessError(RecordingError):
    """Raised when ffmpeg cannot be started or cleanly stopped."""


class RecordingValidationError(RecordingError):
    """Raised when ffprobe rejects a completed recording."""


class RecordingNotFoundError(RecordingError):
    """Raised when a requested completed recording is unavailable."""


@dataclass(slots=True)
class _ActiveRecording:
    profile_id: str
    recording_id: str
    path: Path
    process: subprocess.Popen[bytes]
    stopping: bool = False


@dataclass(frozen=True, slots=True)
class CompletedRecording:
    profile_id: str
    recording_id: str
    path: Path
    duration_s: float


@dataclass(frozen=True, slots=True)
class RecordingSnapshot:
    recording_id: str | None
    state: Literal["recording", "stopped"]


class RecordingStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording_id: str | None
    state: Literal["recording", "stopped"]


class RecordingStopResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording_id: str
    duration_s: float
    path: str


class RecordingManager:
    """Own ffmpeg processes and validated MP4s for running profiles."""

    def __init__(
        self,
        running_profile_getter: Callable[[str], RunningProfile | None],
        *,
        recordings_dir: Path | None = None,
        screen_size_getter: Callable[[str], tuple[int, int] | None] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._running_profile_getter = running_profile_getter
        self._configured_recordings_dir = recordings_dir
        self._screen_size_getter = screen_size_getter
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._active: dict[str, _ActiveRecording] = {}
        self._completed: dict[str, CompletedRecording] = {}
        self._last_by_profile: dict[str, str] = {}
        self._lock = threading.RLock()

    @property
    def recordings_dir(self) -> Path:
        return self._configured_recordings_dir or db.DATA_DIR / "recordings"

    def start(self, profile_id: str) -> RecordingSnapshot:
        self._reap_finished_before_start(profile_id)

        with self._lock:
            if profile_id in self._active:
                raise RecordingConflictError(
                    f"Profile {profile_id} already has an active recording"
                )

            running = self._running_profile_getter(profile_id)
            if running is None:
                raise RecordingUnavailableError(f"Profile {profile_id} is not running")
            display = self._resolve_display(running)
            recording_id, output_path = self._new_recording_path(profile_id)
            output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            command = self._ffmpeg_command(profile_id, display, output_path)

            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                output_path.unlink(missing_ok=True)
                raise RecordingProcessError(
                    "Could not start the screen recorder"
                ) from exc

            self._active[profile_id] = _ActiveRecording(
                profile_id=profile_id,
                recording_id=recording_id,
                path=output_path,
                process=process,
            )
            self._last_by_profile.pop(profile_id, None)

        logger.info("Started recording %s for profile %s", recording_id, profile_id)
        return RecordingSnapshot(recording_id=recording_id, state="recording")

    def stop(self, profile_id: str) -> CompletedRecording:
        with self._lock:
            active = self._active.get(profile_id)
            if active is None:
                raise RecordingNotActiveError(
                    f"Profile {profile_id} has no active recording"
                )
            if active.stopping:
                raise RecordingConflictError(
                    f"Profile {profile_id} recording is already stopping"
                )
            active.stopping = True

        try:
            self._stop_process(active.process)
        except Exception as exc:
            self._discard(active)
            if isinstance(exc, RecordingError):
                raise
            raise RecordingProcessError("Could not stop the screen recorder") from exc

        return self._finalize(active)

    def status(self, profile_id: str) -> RecordingSnapshot:
        with self._lock:
            active = self._active.get(profile_id)
            if active is None:
                return RecordingSnapshot(
                    recording_id=self._last_by_profile.get(profile_id),
                    state="stopped",
                )
            if active.stopping or active.process.poll() is None:
                return RecordingSnapshot(
                    recording_id=active.recording_id,
                    state="recording",
                )
            active.stopping = True

        completed = self._finalize(active)
        return RecordingSnapshot(
            recording_id=completed.recording_id,
            state="stopped",
        )

    def recording_path(self, recording_id: str) -> Path:
        with self._lock:
            completed = self._completed.get(recording_id)
            if completed is None or not completed.path.is_file():
                self._completed.pop(recording_id, None)
                raise RecordingNotFoundError(f"Recording {recording_id} was not found")
            return completed.path

    def _reap_finished_before_start(self, profile_id: str) -> None:
        with self._lock:
            active = self._active.get(profile_id)
            if active is None:
                return
            if active.stopping or active.process.poll() is None:
                raise RecordingConflictError(
                    f"Profile {profile_id} already has an active recording"
                )
            active.stopping = True

        try:
            self._finalize(active)
        except RecordingValidationError:
            logger.warning(
                "Discarded invalid recording %s before starting another",
                active.recording_id,
            )

    def _new_recording_path(self, profile_id: str) -> tuple[str, Path]:
        profile_key = re.sub(r"[^A-Za-z0-9_-]", "_", profile_id).strip("_")
        profile_key = (profile_key or "profile")[:100]
        timestamp = self._now().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        suffix = uuid.uuid4().hex[:8]
        recording_id = f"{profile_key}-{timestamp}-{suffix}"
        path = self.recordings_dir / profile_key / f"{timestamp}-{suffix}.mp4"
        return recording_id, path

    def _ffmpeg_command(
        self,
        profile_id: str,
        display: int,
        output_path: Path,
    ) -> list[str]:
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "x11grab",
            "-framerate",
            str(RECORDING_FPS),
        ]
        if self._screen_size_getter is not None:
            screen_size = self._screen_size_getter(profile_id)
            if screen_size is not None:
                width, height = screen_size
                command.extend(("-video_size", f"{width}x{height}"))
        command.extend(
            (
                "-i",
                f":{display}.0",
                "-t",
                str(MAX_RECORDING_SECONDS),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            )
        )
        return command

    @staticmethod
    def _resolve_display(running: RunningProfile) -> int:
        display = running.display
        if isinstance(display, int) and not isinstance(display, bool) and display >= 0:
            return display

        ws_port = running.ws_port
        if isinstance(ws_port, int) and not isinstance(ws_port, bool):
            offset = ws_port - VNCManager.BASE_WS_PORT
            if offset >= 0:
                return VNCManager.BASE_DISPLAY + offset

        raise RecordingUnavailableError(
            f"Profile {running.profile_id} has no private X display"
        )

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            process.wait(timeout=0)
            return

        wrote_quit = False
        if process.stdin is not None:
            try:
                process.stdin.write(b"q\n")
                process.stdin.flush()
                wrote_quit = True
            except (BrokenPipeError, OSError):
                pass

        if not wrote_quit:
            process.send_signal(signal.SIGINT)

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        finally:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass

    def _finalize(self, active: _ActiveRecording) -> CompletedRecording:
        try:
            duration_s = self._probe_duration(active.path)
        except Exception:
            self._discard(active)
            raise

        completed = CompletedRecording(
            profile_id=active.profile_id,
            recording_id=active.recording_id,
            path=active.path,
            duration_s=duration_s,
        )
        with self._lock:
            if self._active.get(active.profile_id) is active:
                del self._active[active.profile_id]
            self._completed[active.recording_id] = completed
            self._last_by_profile[active.profile_id] = active.recording_id
        logger.info(
            "Finalized recording %s (%.3fs)",
            active.recording_id,
            duration_s,
        )
        return completed

    def _discard(self, active: _ActiveRecording) -> None:
        active.path.unlink(missing_ok=True)
        with self._lock:
            if self._active.get(active.profile_id) is active:
                del self._active[active.profile_id]
            self._last_by_profile.pop(active.profile_id, None)

    @staticmethod
    def _probe_duration(path: Path) -> float:
        if not path.is_file() or path.stat().st_size == 0:
            raise RecordingValidationError("Recording did not produce an MP4")

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RecordingValidationError(
                "Could not validate the completed recording"
            ) from exc

        if result.returncode != 0:
            raise RecordingValidationError("The completed recording is corrupt")
        try:
            payload = json.loads(result.stdout)
            duration_s = float(payload["format"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RecordingValidationError(
                "The completed recording has no valid duration"
            ) from exc
        if not math.isfinite(duration_s) or duration_s < MIN_RECORDING_SECONDS:
            raise RecordingValidationError(
                f"Recording must be at least {MIN_RECORDING_SECONDS} seconds"
            )
        return duration_s


def _running_profile(profile_id: str) -> RunningProfile | None:
    from .main import browser_mgr

    return browser_mgr.running.get(profile_id)


def _profile_screen_size(profile_id: str) -> tuple[int, int] | None:
    profile = db.get_profile(profile_id)
    if profile is None:
        return None
    width = profile.get("screen_width")
    height = profile.get("screen_height")
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
    ):
        return width, height
    return None


recording_manager = RecordingManager(
    _running_profile,
    screen_size_getter=_profile_screen_size,
)


async def on_profile_stopped(profile_id: str) -> CompletedRecording | None:
    """Finalize an active capture before main tears down the profile display."""
    try:
        return await asyncio.to_thread(recording_manager.stop, profile_id)
    except RecordingNotActiveError:
        return None
    except RecordingError as exc:
        logger.warning(
            "Could not finalize recording while profile %s stopped: %s",
            profile_id,
            exc,
        )
        return None


def _recording_http_exception(exc: RecordingError) -> HTTPException:
    if isinstance(exc, RecordingNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, RecordingValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(
        exc,
        (RecordingConflictError, RecordingNotActiveError, RecordingUnavailableError),
    ):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/api/profiles/{profile_id}/recording/start",
    response_model=RecordingStatusResponse,
)
async def start_recording(profile_id: str) -> RecordingStatusResponse:
    try:
        snapshot = await asyncio.to_thread(recording_manager.start, profile_id)
    except RecordingError as exc:
        raise _recording_http_exception(exc) from exc
    return RecordingStatusResponse(
        recording_id=snapshot.recording_id,
        state=snapshot.state,
    )


@router.post(
    "/api/profiles/{profile_id}/recording/stop",
    response_model=RecordingStopResponse,
)
async def stop_recording(profile_id: str) -> RecordingStopResponse:
    try:
        completed = await asyncio.to_thread(recording_manager.stop, profile_id)
    except RecordingError as exc:
        raise _recording_http_exception(exc) from exc
    return RecordingStopResponse(
        recording_id=completed.recording_id,
        duration_s=completed.duration_s,
        path=f"/api/recording/{completed.recording_id}",
    )


@router.get(
    "/api/profiles/{profile_id}/recording/status",
    response_model=RecordingStatusResponse,
)
async def recording_status(profile_id: str) -> RecordingStatusResponse:
    try:
        snapshot = await asyncio.to_thread(recording_manager.status, profile_id)
    except RecordingError as exc:
        raise _recording_http_exception(exc) from exc
    return RecordingStatusResponse(
        recording_id=snapshot.recording_id,
        state=snapshot.state,
    )


@router.get("/api/recording/{recording_id}", response_class=FileResponse)
async def download_recording(recording_id: str) -> FileResponse:
    try:
        path = recording_manager.recording_path(recording_id)
    except RecordingError as exc:
        raise _recording_http_exception(exc) from exc
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{recording_id}.mp4",
    )
