"""Tests for per-profile X display recording and its HTTP contract."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from backend import recorder
from backend.browser_manager import RunningProfile
from backend.recorder import (
    RecordingConflictError,
    RecordingManager,
    RecordingUnavailableError,
    RecordingValidationError,
)

PROFILE_ID = "profile-1"


@pytest.fixture()
def recording_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = 0
    process.stdin = MagicMock()

    popen = MagicMock()

    def spawn(command: list[str], **kwargs):
        Path(command[-1]).write_bytes(b"mock mp4")
        return process

    popen.side_effect = spawn
    probe = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps({"format": {"duration": "8.25"}}),
            stderr="",
        )
    )
    monkeypatch.setattr(recorder.subprocess, "Popen", popen)
    monkeypatch.setattr(recorder.subprocess, "run", probe)

    running = RunningProfile(
        profile_id=PROFILE_ID,
        context=object(),
        cdp_port=9222,
        display=None,
        ws_port=6103,
    )
    manager = RecordingManager(
        lambda profile_id: running if profile_id == PROFILE_ID else None,
        recordings_dir=tmp_path / "recordings",
        screen_size_getter=lambda _profile_id: (1280, 720),
        now=lambda: datetime(2026, 9, 12, 12, 30, tzinfo=timezone.utc),
    )
    return manager, popen, probe, process


def test_recording_lifecycle_spawns_expected_ffmpeg_and_finalizes(
    recording_fixture,
) -> None:
    manager, popen, probe, process = recording_fixture

    started = manager.start(PROFILE_ID)

    assert started.state == "recording"
    assert started.recording_id is not None
    command = popen.call_args.args[0]
    assert command[command.index("-f") + 1] == "x11grab"
    assert command[command.index("-framerate") + 1] == "15"
    assert command[command.index("-video_size") + 1] == "1280x720"
    assert command[command.index("-i") + 1] == ":103.0"
    assert command[command.index("-t") + 1] == "600"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert "-an" in command
    assert popen.call_args.kwargs == {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    assert manager.status(PROFILE_ID) == started

    completed = manager.stop(PROFILE_ID)

    process.stdin.write.assert_called_once_with(b"q\n")
    process.stdin.flush.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=10)
    assert completed.recording_id == started.recording_id
    assert completed.duration_s == pytest.approx(8.25)
    assert completed.path.parent.name == PROFILE_ID
    assert manager.recording_path(completed.recording_id) == completed.path
    assert manager.status(PROFILE_ID).state == "stopped"
    assert probe.call_args.args[0][0] == "ffprobe"


def test_only_one_recording_can_be_active_per_profile(recording_fixture) -> None:
    manager, popen, _probe, _process = recording_fixture
    manager.start(PROFILE_ID)

    with pytest.raises(RecordingConflictError, match="active recording"):
        manager.start(PROFILE_ID)

    popen.assert_called_once()


def test_recording_requires_a_running_private_display(tmp_path: Path) -> None:
    native = RunningProfile(
        profile_id=PROFILE_ID,
        context=object(),
        cdp_port=9222,
        display=None,
        ws_port=None,
    )
    manager = RecordingManager(
        lambda _profile_id: native,
        recordings_dir=tmp_path / "recordings",
    )

    with pytest.raises(RecordingUnavailableError, match="no private X display"):
        manager.start(PROFILE_ID)


@pytest.mark.parametrize(
    ("probe_result", "message"),
    [
        (
            subprocess.CompletedProcess(
                args=["ffprobe"],
                returncode=1,
                stdout="",
                stderr="invalid data",
            ),
            "corrupt",
        ),
        (
            subprocess.CompletedProcess(
                args=["ffprobe"],
                returncode=0,
                stdout=json.dumps({"format": {"duration": "0.49"}}),
                stderr="",
            ),
            "at least 0.5 seconds",
        ),
    ],
)
def test_validation_rejects_and_deletes_unusable_recordings(
    recording_fixture,
    probe_result: subprocess.CompletedProcess[str],
    message: str,
) -> None:
    manager, _popen, probe, _process = recording_fixture
    manager.start(PROFILE_ID)
    probe.return_value = probe_result

    with pytest.raises(RecordingValidationError, match=message):
        manager.stop(PROFILE_ID)

    assert list(manager.recordings_dir.rglob("*.mp4")) == []
    assert manager.status(PROFILE_ID).recording_id is None


async def test_profile_stop_hook_finalizes_an_active_recording(
    recording_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _popen, _probe, process = recording_fixture
    monkeypatch.setattr(recorder, "recording_manager", manager)
    started = manager.start(PROFILE_ID)

    completed = await recorder.on_profile_stopped(PROFILE_ID)

    assert completed is not None
    assert completed.recording_id == started.recording_id
    process.stdin.write.assert_called_once_with(b"q\n")
    assert manager.status(PROFILE_ID).state == "stopped"
    assert await recorder.on_profile_stopped(PROFILE_ID) is None


def test_recording_endpoints_match_client_contract(
    app_client: TestClient,
    recording_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _popen, _probe, _process = recording_fixture
    monkeypatch.setattr(recorder, "recording_manager", manager)

    idle = app_client.get(f"/api/profiles/{PROFILE_ID}/recording/status")
    assert idle.status_code == 200
    assert idle.json() == {"recording_id": None, "state": "stopped"}

    started = app_client.post(f"/api/profiles/{PROFILE_ID}/recording/start")
    assert started.status_code == 200
    assert started.json()["state"] == "recording"
    recording_id = started.json()["recording_id"]

    duplicate = app_client.post(f"/api/profiles/{PROFILE_ID}/recording/start")
    assert duplicate.status_code == 409
    assert app_client.get(f"/api/profiles/{PROFILE_ID}/recording/status").json() == {
        "recording_id": recording_id,
        "state": "recording",
    }

    stopped = app_client.post(f"/api/profiles/{PROFILE_ID}/recording/stop")
    assert stopped.status_code == 200
    assert stopped.json() == {
        "recording_id": recording_id,
        "duration_s": 8.25,
        "path": f"/api/recording/{recording_id}",
    }

    downloaded = app_client.get(f"/api/recording/{recording_id}")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "video/mp4"
    assert downloaded.content == b"mock mp4"

    missing = app_client.get("/api/recording/not-a-recording")
    assert missing.status_code == 404


def test_profile_stop_finalizes_recording_before_display_teardown(
    app_client: TestClient,
    recording_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend import main

    manager, _popen, _probe, process = recording_fixture
    monkeypatch.setattr(recorder, "recording_manager", manager)
    running = manager._running_profile_getter(PROFILE_ID)
    main.browser_mgr.running[PROFILE_ID] = running
    browser_stop = MagicMock()

    async def stop_profile(profile_id: str) -> None:
        browser_stop(profile_id)
        assert process.stdin.write.call_count == 1
        main.browser_mgr.running.pop(profile_id, None)

    monkeypatch.setattr(main.browser_mgr, "stop", stop_profile)
    manager.start(PROFILE_ID)

    response = app_client.post(f"/api/profiles/{PROFILE_ID}/stop")

    assert response.status_code == 200
    browser_stop.assert_called_once_with(PROFILE_ID)
    assert manager.status(PROFILE_ID).state == "stopped"
