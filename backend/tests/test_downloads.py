"""Tests for capturing files the profile's browser downloads."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from backend import artifacts
from backend import database as db
from backend import downloads


@pytest.fixture()
def profile(tmp_db: Path) -> dict:
    return db.create_profile(name="Downloader")


def _stage(profile_id: str, guid: str, payload: bytes) -> Path:
    """Stand in for Chromium writing a GUID-named file into the staging directory."""
    staging = downloads.staging_dir(profile_id)
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / guid
    path.write_bytes(payload)
    return path


def test_a_started_download_is_recorded_as_pending(profile: dict):
    tracker = downloads.DownloadTracker(profile["id"])
    artifact_id = tracker.begin("guid-1", "weekly report.csv")

    row = db.get_artifact(profile["id"], artifact_id)
    assert row["state"] == "pending"
    assert row["kind"] == "download"
    assert row["name"] == "weekly report.csv"
    assert row["size"] == 0


def test_a_suggested_filename_cannot_escape_the_artifact_directory(profile: dict):
    tracker = downloads.DownloadTracker(profile["id"])
    artifact_id = tracker.begin("guid-1", "../../../../etc/passwd")
    assert db.get_artifact(profile["id"], artifact_id)["name"] == "passwd"


def test_a_nameless_download_still_gets_a_name(profile: dict):
    tracker = downloads.DownloadTracker(profile["id"])
    artifact_id = tracker.begin("guid-1", None)
    assert db.get_artifact(profile["id"], artifact_id)["name"] == "download"


def test_a_completed_download_is_published_with_its_real_size(profile: dict):
    tracker = downloads.DownloadTracker(profile["id"])
    artifact_id = tracker.begin("guid-1", "orders.csv")
    _stage(profile["id"], "guid-1", b"store_id,orders\n1,2\n")

    tracker.progress("guid-1", "completed", received_bytes=999)

    row = db.get_artifact(profile["id"], artifact_id)
    assert row["state"] == "ready"
    # Size comes from the bytes on disk, not from the browser's claim.
    assert row["size"] == len(b"store_id,orders\n1,2\n")
    stored = artifacts.artifact_path(profile["id"], artifact_id, "orders.csv")
    assert stored.read_bytes() == b"store_id,orders\n1,2\n"
    # The staging copy is gone; the file lives at the canonical path only.
    assert not (downloads.staging_dir(profile["id"]) / "guid-1").exists()


def test_an_in_progress_download_is_not_published(profile: dict):
    tracker = downloads.DownloadTracker(profile["id"])
    artifact_id = tracker.begin("guid-1", "big.csv")
    _stage(profile["id"], "guid-1", b"partial")

    tracker.progress("guid-1", "inProgress", received_bytes=7)

    assert db.get_artifact(profile["id"], artifact_id)["state"] == "pending"
    assert not artifacts.artifact_path(profile["id"], artifact_id, "big.csv").exists()


def test_completed_without_the_file_is_a_failure_not_an_empty_artifact(profile: dict):
    """CDP's completed event does not promise the file is there."""
    tracker = downloads.DownloadTracker(profile["id"])
    artifact_id = tracker.begin("guid-1", "ghost.csv")

    tracker.progress("guid-1", "completed", received_bytes=10)

    row = db.get_artifact(profile["id"], artifact_id)
    assert row["state"] == "failed"
    assert not artifacts.artifact_path(profile["id"], artifact_id, "ghost.csv").exists()


def test_a_cancelled_download_leaves_nothing_behind(profile: dict):
    tracker = downloads.DownloadTracker(profile["id"])
    artifact_id = tracker.begin("guid-1", "aborted.csv")
    _stage(profile["id"], "guid-1", b"half")

    tracker.progress("guid-1", "canceled", received_bytes=4)

    assert db.get_artifact(profile["id"], artifact_id) is None
    assert not (downloads.staging_dir(profile["id"]) / "guid-1").exists()
    assert db.list_artifacts(profile["id"]) == []


def test_progress_for_an_unknown_download_is_ignored(profile: dict):
    downloads.DownloadTracker(profile["id"]).progress("never-seen", "completed", 1)
    assert db.list_artifacts(profile["id"]) == []


def test_downloads_are_refused_once_the_profile_is_at_its_cap(
    profile: dict, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(artifacts, "MAX_ARTIFACTS_PER_PROFILE", 1)
    tracker = downloads.DownloadTracker(profile["id"])
    assert tracker.begin("guid-1", "one.csv") is not None
    assert tracker.begin("guid-2", "two.csv") is None
    assert len(db.list_artifacts(profile["id"])) == 1


def test_a_published_download_shows_up_in_the_file_chooser_view(profile: dict):
    tracker = downloads.DownloadTracker(profile["id"])
    tracker.begin("guid-1", "payout.csv")
    _stage(profile["id"], "guid-1", b"x")
    tracker.progress("guid-1", "completed", 1)

    link = artifacts.picker_dir(profile["id"]) / "payout.csv"
    assert link.is_symlink() and link.read_bytes() == b"x"


def test_handle_event_routes_cdp_messages(profile: dict):
    tracker = downloads.DownloadTracker(profile["id"])
    downloads.handle_event(tracker, {
        "method": "Browser.downloadWillBegin",
        "params": {"guid": "g", "suggestedFilename": "report.xlsx", "url": "https://x/y"},
    })
    assert [row["state"] for row in db.list_artifacts(profile["id"])] == ["pending"]

    _stage(profile["id"], "g", b"sheet")
    downloads.handle_event(tracker, {
        "method": "Browser.downloadProgress",
        "params": {"guid": "g", "state": "completed", "receivedBytes": 5, "totalBytes": 5},
    })
    row = db.list_artifacts(profile["id"])[0]
    assert (row["state"], row["size"], row["name"]) == ("ready", 5, "report.xlsx")


def test_handle_event_ignores_unrelated_methods(profile: dict):
    downloads.handle_event(downloads.DownloadTracker(profile["id"]), {"method": "Page.loadEventFired"})
    assert db.list_artifacts(profile["id"]) == []


# ── API surface ──────────────────────────────────────────────────────────────


def test_a_download_in_flight_is_listed_but_not_servable(app_client: TestClient, tmp_db: Path):
    pid = app_client.post("/api/profiles", json={"name": "Downloader"}).json()["id"]
    artifact_id = downloads.DownloadTracker(pid).begin("guid-1", "in-flight.csv")

    listed = app_client.get(f"/api/profiles/{pid}/files").json()
    assert [(a["name"], a["state"], a["kind"]) for a in listed] == [("in-flight.csv", "pending", "download")]

    resp = app_client.get(f"/api/profiles/{pid}/files/{artifact_id}")
    assert resp.status_code == 409
    assert "still in progress" in resp.json()["detail"]
