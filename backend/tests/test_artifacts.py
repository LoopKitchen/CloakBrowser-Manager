"""Tests for per-profile file artifacts (upload / list / download / delete)."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from backend import artifacts
from backend import database as db


def _upload(client: TestClient, pid: str, name: str = "report.csv", data: bytes = b"a,b\n1,2\n"):
    return client.post(f"/api/profiles/{pid}/files", files={"file": (name, data, "text/csv")})


def _new_profile(client: TestClient, name: str = "Files") -> str:
    return client.post("/api/profiles", json={"name": name}).json()["id"]


# ── safe_filename ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("report.csv", "report.csv"),
    ("../../etc/passwd", "passwd"),
    ("..\\..\\windows\\system32\\cfg", "cfg"),
    ("/absolute/path.xlsx", "path.xlsx"),
    ("", "file"),
    ("..", "file"),
    (".", "file"),
    (None, "file"),
    ("with\x00null.csv", "withnull.csv"),
])
def test_safe_filename_neutralises_paths(raw, expected):
    assert artifacts.safe_filename(raw) == expected


def test_safe_filename_caps_length():
    assert len(artifacts.safe_filename("x" * 500)) == 255


# ── upload ───────────────────────────────────────────────────────────────────


def test_upload_stores_the_file_and_reports_its_container_path(app_client: TestClient, tmp_db: Path):
    pid = _new_profile(app_client)
    resp = _upload(app_client, pid, "weekly report.csv", b"payload")
    assert resp.status_code == 201
    art = resp.json()
    assert art["name"] == "weekly report.csv"
    assert art["size"] == len("payload")
    assert art["kind"] == "upload" and art["state"] == "ready"
    # The page is handed the real filename, so a portal that checks the extension accepts it.
    path = Path(art["container_path"])
    assert path.name == "weekly report.csv"
    assert path.read_bytes() == b"payload"


def test_uploads_live_outside_the_profile_user_data_dir(app_client: TestClient, tmp_db: Path):
    """The whole point of a separate artifact root: browser-state copies must not carry these."""
    pid = _new_profile(app_client)
    art = _upload(app_client, pid).json()
    user_data_dir = Path(app_client.get(f"/api/profiles/{pid}").json()["user_data_dir"])
    assert user_data_dir not in Path(art["container_path"]).parents


def test_upload_rejects_unknown_profile(app_client: TestClient):
    assert _upload(app_client, "nonexistent").status_code == 404


def test_upload_filename_cannot_escape_the_artifact_directory(app_client: TestClient, tmp_db: Path):
    pid = _new_profile(app_client)
    art = _upload(app_client, pid, "../../../../etc/passwd", b"nope").json()
    stored = Path(art["container_path"])
    assert stored.name == "passwd"
    assert artifacts.artifacts_root() in stored.parents
    assert stored.read_bytes() == b"nope"


def test_upload_over_the_size_cap_is_rejected_and_leaves_nothing(
    app_client: TestClient, tmp_db: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(artifacts, "MAX_ARTIFACT_BYTES", 8)
    pid = _new_profile(app_client)
    assert _upload(app_client, pid, "big.csv", b"x" * 64).status_code == 413
    assert app_client.get(f"/api/profiles/{pid}/files").json() == []
    leftovers = list(artifacts.artifact_dir(pid).rglob("*")) if artifacts.artifact_dir(pid).exists() else []
    assert [p for p in leftovers if p.is_file()] == []


def test_upload_over_the_count_quota_is_rejected(
    app_client: TestClient, tmp_db: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(artifacts, "MAX_ARTIFACTS_PER_PROFILE", 1)
    pid = _new_profile(app_client)
    assert _upload(app_client, pid, "one.csv").status_code == 201
    assert _upload(app_client, pid, "two.csv").status_code == 507


# ── list / download / delete ─────────────────────────────────────────────────


def test_list_and_download_round_trip(app_client: TestClient, tmp_db: Path):
    pid = _new_profile(app_client)
    payload = b"store_id,orders\n23197600,142\n"
    art = _upload(app_client, pid, "weekly report.csv", payload).json()

    listing = app_client.get(f"/api/profiles/{pid}/files").json()
    assert [a["id"] for a in listing] == [art["id"]]

    resp = app_client.get(f"/api/profiles/{pid}/files/{art['id']}")
    assert resp.status_code == 200
    assert resp.content == payload
    # Merchant documents are never rendered on the Manager's own origin.
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["content-type"] == "application/octet-stream"
    assert "attachment" in resp.headers["content-disposition"]
    assert "weekly%20report.csv" in resp.headers["content-disposition"]


def test_download_unknown_artifact_404(app_client: TestClient, tmp_db: Path):
    pid = _new_profile(app_client)
    assert app_client.get(f"/api/profiles/{pid}/files/nope").status_code == 404


def test_artifact_of_another_profile_is_not_reachable(app_client: TestClient, tmp_db: Path):
    owner, other = _new_profile(app_client, "Owner"), _new_profile(app_client, "Other")
    art = _upload(app_client, owner).json()
    assert app_client.get(f"/api/profiles/{other}/files/{art['id']}").status_code == 404
    assert app_client.delete(f"/api/profiles/{other}/files/{art['id']}").status_code == 404


def test_delete_removes_row_and_bytes(app_client: TestClient, tmp_db: Path):
    pid = _new_profile(app_client)
    art = _upload(app_client, pid).json()
    assert app_client.delete(f"/api/profiles/{pid}/files/{art['id']}").status_code == 200
    assert app_client.get(f"/api/profiles/{pid}/files").json() == []
    assert app_client.get(f"/api/profiles/{pid}/files/{art['id']}").status_code == 404
    assert not Path(art["container_path"]).exists()
    assert app_client.delete(f"/api/profiles/{pid}/files/{art['id']}").status_code == 404


# ── lifecycle interactions ───────────────────────────────────────────────────


def test_deleting_a_profile_removes_its_artifacts(app_client: TestClient, tmp_db: Path):
    pid = _new_profile(app_client)
    art = _upload(app_client, pid).json()
    assert app_client.delete(f"/api/profiles/{pid}").status_code == 200
    assert not Path(art["container_path"]).exists()
    assert not artifacts.artifact_dir(pid).exists()
    assert db.list_artifacts(pid) == []


def test_duplicate_with_browser_state_does_not_carry_artifacts(app_client: TestClient, tmp_db: Path):
    """Merchant documents must not be silently cloned into a copy of a profile."""
    pid = _new_profile(app_client, "Src")
    art = _upload(app_client, pid, "confidential.csv", b"secret").json()
    user_data_dir = Path(app_client.get(f"/api/profiles/{pid}").json()["user_data_dir"])
    (user_data_dir / "Default").mkdir(parents=True)
    (user_data_dir / "Default" / "Cookies").write_text("session=abc")

    clone = app_client.post(
        f"/api/profiles/{pid}/duplicate", json={"include_browser_state": True}
    ).json()

    # The session travels with the clone; the merchant's report does not.
    assert (Path(clone["user_data_dir"]) / "Default" / "Cookies").read_text() == "session=abc"
    assert app_client.get(f"/api/profiles/{clone['id']}/files").json() == []
    assert not artifacts.artifact_dir(clone["id"]).exists()
    assert Path(art["container_path"]).read_bytes() == b"secret"
