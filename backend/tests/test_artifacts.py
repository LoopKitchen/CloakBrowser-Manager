"""Tests for per-profile file artifacts (upload / list / download / delete)."""

from __future__ import annotations

from pathlib import Path

import pytest
from unittest.mock import MagicMock
from starlette.testclient import TestClient

from backend import artifacts
from backend import database as db
from backend import main


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


# ── file-chooser convenience view ────────────────────────────────────────────


def test_upload_appears_under_its_real_name_in_the_picker_view(app_client: TestClient, tmp_db: Path, docker_runtime):
    """A page's native 'Choose File' dialog must be able to reach the file by name."""
    pid = _new_profile(app_client)
    art = _upload(app_client, pid, "weekly report.csv", b"payload").json()
    link = artifacts.picker_dir(pid) / "weekly report.csv"
    assert link.is_symlink()
    assert link.resolve() == Path(art["container_path"]).resolve()
    assert link.read_bytes() == b"payload"


def test_picker_view_disambiguates_duplicate_names(app_client: TestClient, tmp_db: Path, docker_runtime):
    pid = _new_profile(app_client)
    _upload(app_client, pid, "report.csv", b"one")
    _upload(app_client, pid, "report.csv", b"two")
    names = sorted(p.name for p in artifacts.picker_dir(pid).iterdir())
    assert names == ["report (1).csv", "report.csv"]


def test_picker_view_drops_a_deleted_file(app_client: TestClient, tmp_db: Path, docker_runtime):
    pid = _new_profile(app_client)
    art = _upload(app_client, pid, "gone.csv").json()
    app_client.delete(f"/api/profiles/{pid}/files/{art['id']}")
    assert list(artifacts.picker_dir(pid).iterdir()) == []


def test_the_file_chooser_view_is_docker_only(
    app_client: TestClient, tmp_db: Path, native_runtime
):
    """On a native macOS/Windows install the browser already sees the user's own disk, so
    the view is unnecessary — and creating symlinks there needs a Windows privilege."""
    assert artifacts.picker_view_enabled() is False
    assert artifacts._gtk_bookmarks_path() is None

    pid = _new_profile(app_client)
    art = _upload(app_client, pid, "report.csv", b"payload").json()

    # The file itself is stored and served exactly as on Linux...
    assert Path(art["container_path"]).read_bytes() == b"payload"
    assert app_client.get(f"/api/profiles/{pid}/files/{art['id']}").content == b"payload"
    # ...but no symlink view and no bookmarks file are created.
    assert not artifacts.picker_dir(pid).exists()


def test_gtk_bookmark_points_at_the_picker_view(
    app_client: TestClient, tmp_db: Path, docker_runtime
):
    bookmarks = artifacts._gtk_bookmarks_path()
    assert bookmarks is not None
    pid = _new_profile(app_client, "Work profile")
    _upload(app_client, pid, "menu.csv")
    line = bookmarks.read_text().strip()
    assert line.startswith(f"file://{artifacts.picker_dir(pid)} ")
    assert line.endswith("Work profile")


def test_a_picker_name_keeps_pointing_at_the_document_it_was_given_to(
    app_client: TestClient, tmp_db: Path, docker_runtime):
    """Two files sharing a name must not swap identities when the view is rebuilt."""
    pid = _new_profile(app_client)
    first = _upload(app_client, pid, "report.csv", b"FIRST").json()
    second = _upload(app_client, pid, "report.csv", b"SECOND").json()

    assert first["picker_name"] == "report.csv"
    assert second["picker_name"] == "report (1).csv"
    view = artifacts.picker_dir(pid)
    # The name the first upload was given still resolves to the first upload's bytes.
    assert (view / "report.csv").read_bytes() == b"FIRST"
    assert (view / "report (1).csv").read_bytes() == b"SECOND"

    # ...and still does after an unrelated change forces another sync.
    _upload(app_client, pid, "other.csv", b"OTHER")
    assert (view / "report.csv").read_bytes() == b"FIRST"


def test_deleting_a_file_does_not_rename_its_neighbours(app_client: TestClient, tmp_db: Path, docker_runtime):
    pid = _new_profile(app_client)
    first = _upload(app_client, pid, "report.csv", b"FIRST").json()
    second = _upload(app_client, pid, "report.csv", b"SECOND").json()

    app_client.delete(f"/api/profiles/{pid}/files/{first['id']}")

    view = artifacts.picker_dir(pid)
    assert not (view / "report.csv").exists()
    # The survivor keeps the name it was given rather than being promoted onto the free one.
    assert (view / "report (1).csv").read_bytes() == b"SECOND"
    assert db.get_artifact(pid, second["id"])["picker_name"] == "report (1).csv"


def test_safe_filename_fits_the_filesystem_byte_limit_and_keeps_the_extension():
    # Three bytes per character, so 200 characters is 600 bytes — well over the 255-byte cap.
    name = ("長" * 200) + ".csv"
    fitted = artifacts.safe_filename(name)
    assert len(fitted.encode("utf-8")) <= 255
    assert fitted.endswith(".csv")


def test_only_profiles_holding_files_are_reconciled(app_client: TestClient, tmp_db: Path):
    """The startup repair must not walk every profile on a fleet-sized Manager."""
    with_files = _new_profile(app_client, "Has files")
    _new_profile(app_client, "Empty")
    _upload(app_client, with_files)
    assert db.profile_ids_with_artifacts() == [with_files]


def test_startup_repair_drops_a_picker_link_whose_artifact_is_gone(
    app_client: TestClient, tmp_db: Path, docker_runtime):
    pid = _new_profile(app_client)
    _upload(app_client, pid, "kept.csv")
    stale = artifacts.picker_dir(pid) / "vanished.csv"
    stale.symlink_to(artifacts.artifact_dir(pid) / "nowhere" / "vanished.csv")

    artifacts.sync_picker_view(pid)

    assert not stale.is_symlink()
    assert (artifacts.picker_dir(pid) / "kept.csv").is_symlink()


# ── cross-platform filenames ─────────────────────────────────────────────────
# The Manager ships as a Linux container AND as native macOS/Windows apps, and a /data
# directory written by one can be opened by another, so names must satisfy all of them.


@pytest.mark.parametrize("raw,expected", [
    ('report:2026.csv', "report_2026.csv"),      # ':' is illegal on Windows
    ('a<b>c|d?e*f".csv', "a_b_c_d_e_f_.csv"),
    ("trailing dot..", "trailing dot"),          # Windows silently drops these
    ("trailing space  ", "trailing space"),
    ("NUL", "_NUL"),                             # reserved device names
    ("con.txt", "_con.txt"),
    ("COM1.csv", "_COM1.csv"),
    ("console.csv", "console.csv"),              # only the exact name is reserved
])
def test_safe_filename_satisfies_windows_rules_too(raw, expected):
    assert artifacts.safe_filename(raw) == expected


def test_a_maximum_length_duplicate_name_still_resolves(app_client: TestClient, tmp_db: Path, docker_runtime):
    """Trimming must never regenerate the taken name, or the rename loop never terminates."""
    pid = _new_profile(app_client)
    longest = ("a" * 251) + ".csv"
    assert len(longest.encode()) == 255

    first = _upload(app_client, pid, longest, b"FIRST").json()
    second = _upload(app_client, pid, longest, b"SECOND").json()

    assert first["picker_name"] != second["picker_name"]
    for art in (first, second):
        assert len(art["picker_name"].encode()) <= 255
        assert art["picker_name"].endswith(".csv")
    view = artifacts.picker_dir(pid)
    assert (view / first["picker_name"]).read_bytes() == b"FIRST"
    assert (view / second["picker_name"]).read_bytes() == b"SECOND"


@pytest.mark.parametrize("raw", ["x." + "a" * 300, "x." + "長" * 100, "長" * 400])
def test_safe_filename_never_exceeds_the_byte_budget(raw):
    """An extension can be longer than the whole budget on its own."""
    assert len(artifacts.safe_filename(raw).encode("utf-8")) <= 255


@pytest.mark.parametrize("raw", ["COM¹", "LPT²", "com³"])
def test_safe_filename_covers_the_superscript_device_names(raw):
    assert artifacts.safe_filename(raw).startswith("_")


def test_an_upload_that_cannot_be_recorded_leaves_no_file_behind(
    app_client: TestClient, tmp_db: Path, monkeypatch: pytest.MonkeyPatch
):
    """Bytes with no row could never be listed, deleted, or counted against the quota."""
    pid = _new_profile(app_client)
    monkeypatch.setattr(db, "create_artifact", MagicMock(side_effect=RuntimeError("disk full")))

    resp = _upload(app_client, pid, "doomed.csv", b"payload")

    assert resp.status_code == 500
    assert app_client.get(f"/api/profiles/{pid}/files").json() == []
    leftovers = [p for p in artifacts.artifact_dir(pid).rglob("*") if p.is_file()]
    assert leftovers == []


# ── receipt bound ────────────────────────────────────────────────────────────
# The per-file cap bounds what is STORED. FastAPI parses the whole multipart body before
# the endpoint runs, so the body has to be bounded at the socket as well.


def test_an_oversized_upload_is_refused_without_reaching_the_endpoint(
    app_client: TestClient, tmp_db: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(artifacts, "MAX_ARTIFACT_BYTES", 1024)
    monkeypatch.setattr(artifacts, "save_stream", MagicMock(side_effect=AssertionError("reached")))
    pid = _new_profile(app_client)

    resp = _upload(app_client, pid, "huge.csv", b"x" * (256 * 1024))

    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"]
    assert app_client.get(f"/api/profiles/{pid}/files").json() == []


@pytest.mark.asyncio
async def test_body_limit_stops_a_body_that_understates_its_length(
    monkeypatch: pytest.MonkeyPatch,
):
    """A client may omit or lie about Content-Length, so arriving bytes are counted."""
    monkeypatch.setattr(artifacts, "MAX_ARTIFACT_BYTES", 16)
    chunks = iter([
        {"type": "http.request", "body": b"x" * 40_000, "more_body": True},
        {"type": "http.request", "body": b"x" * 40_000, "more_body": True},
        {"type": "http.request", "body": b"", "more_body": False},
    ])
    consumed = 0

    async def downstream(scope, receive, send):
        nonlocal consumed
        while True:
            message = await receive()
            consumed += len(message.get("body") or b"")
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    sent: list[dict] = []

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    await main.BodyLimitMiddleware(downstream)(
        {"type": "http", "method": "POST", "path": "/api/profiles/p1/files", "headers": []},
        receive, send,
    )

    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
    assert sent[0]["status"] == 413
    # The parser was cut off rather than being fed the whole body.
    assert consumed <= artifacts.MAX_ARTIFACT_BYTES + 64 * 1024 + 40_000


# ── reclaiming what a hard kill left ─────────────────────────────────────────


def test_startup_reclaims_bytes_with_no_row(app_client: TestClient, tmp_db: Path):
    """A container kill never runs the endpoint's cleanup."""
    pid = _new_profile(app_client)
    kept = _upload(app_client, pid, "kept.csv", b"keep").json()

    orphan_dir = artifacts.artifact_dir(pid) / "11111111-2222-3333-4444-555555555555"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "ghost.csv").write_bytes(b"orphan")
    half_written = artifacts.artifact_dir(pid) / ".incoming-99999999"
    half_written.write_bytes(b"partial")

    assert artifacts.reclaim_all() == 2

    assert not orphan_dir.exists()
    assert not half_written.exists()
    assert Path(kept["container_path"]).read_bytes() == b"keep"
    assert [a["id"] for a in app_client.get(f"/api/profiles/{pid}/files").json()] == [kept["id"]]


def test_reclaim_leaves_the_picker_view_and_staging_alone(
    app_client: TestClient, tmp_db: Path, docker_runtime
):
    pid = _new_profile(app_client)
    _upload(app_client, pid, "kept.csv", b"keep")
    staging = artifacts.artifact_dir(pid) / artifacts.STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)

    assert artifacts.reclaim_all() == 0
    assert artifacts.picker_dir(pid).is_dir()
    assert staging.is_dir()


def test_renaming_a_profile_relabels_its_chooser_entry(
    app_client: TestClient, tmp_db: Path, docker_runtime
):
    pid = _new_profile(app_client, "before-rename")
    _upload(app_client, pid, "report.csv")
    bookmarks = artifacts._gtk_bookmarks_path()
    assert bookmarks is not None and "before-rename" in bookmarks.read_text()

    app_client.put(f"/api/profiles/{pid}", json={"name": "after-rename"})

    assert "after-rename" in bookmarks.read_text()
    assert "before-rename" not in bookmarks.read_text()


def test_bookmarks_are_built_without_hydrating_every_profile(
    app_client: TestClient, tmp_db: Path, docker_runtime, monkeypatch: pytest.MonkeyPatch
):
    """list_profiles runs a tag query per profile; this rebuilds on every file mutation."""
    monkeypatch.setattr(db, "list_profiles", MagicMock(side_effect=AssertionError("hydrated")))
    pid = _new_profile(app_client, "dd-merchant")

    _upload(app_client, pid, "report.csv")

    assert "dd-merchant" in artifacts._gtk_bookmarks_path().read_text()
