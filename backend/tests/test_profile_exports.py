import io
import json
import zipfile
from pathlib import Path

import pytest

from backend import database as db


@pytest.fixture(autouse=True)
def authenticated_exports(app_client, monkeypatch):
    from backend import main

    monkeypatch.setattr(main, "AUTH_TOKEN", "local-export-test")
    app_client.headers["Authorization"] = "Bearer local-export-test"


def seed_state(profile):
    root = Path(profile["user_data_dir"])
    (root / "Default" / "Local Storage").mkdir(parents=True)
    (root / "Local State").write_text("browser-key-state")
    (root / "Default" / "Cookies").write_bytes(b"cookie-db")
    (root / "Default" / "Local Storage" / "storage").write_bytes(b"local-storage")
    (root / "SingletonLock").write_text("old-lock")
    return root


def test_export_preserves_storage_and_safe_launch_manifest(app_client, sample_profile):
    root = seed_state(sample_profile)
    db.update_profile(
        sample_profile["id"],
        proxy="http://user:secret@proxy:123",
        notes="secret note",
        launch_args=[],
    )
    r = app_client.post(f"/api/profiles/{sample_profile['id']}/export")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert z.read("user-data/Default/Cookies") == b"cookie-db"
        assert z.read("user-data/Local State") == b"browser-key-state"
        assert "user-data/SingletonLock" not in z.namelist()
        m = json.loads(z.read("profile.json"))
        assert m["format"] == "cloak-profile-v1"
        assert m["fingerprint_seed"] == 12345
        assert "--fingerprint=12345" in m["fingerprint_args"]
        assert "--fingerprint-platform=windows" in m["fingerprint_args"]
        assert m["host_os"] == "linux"
        assert "secret" not in json.dumps(m)
        assert "proxy" not in m and "notes" not in m and "launch_args" not in m
    assert (root / "Default" / "Cookies").read_bytes() == b"cookie-db"


def test_export_refuses_missing_empty_and_active_profiles(app_client, sample_profile, monkeypatch):
    from backend import main

    assert app_client.post("/api/profiles/missing/export").status_code == 404
    assert app_client.post(f"/api/profiles/{sample_profile['id']}/export").status_code == 409
    seed_state(sample_profile)
    monkeypatch.setattr(main.browser_mgr, "is_active", lambda _: True)
    assert app_client.post(f"/api/profiles/{sample_profile['id']}/export").status_code == 409


def test_export_never_follows_symlinks(app_client, sample_profile, tmp_path):
    root = seed_state(sample_profile)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("do-not-export")
    try:
        (root / "outside").symlink_to(outside, target_is_directory=True)
        (root / "secret-link").symlink_to(outside / "secret")
    except OSError:
        pytest.skip("symlinks not available")
    r = app_client.post(f"/api/profiles/{sample_profile['id']}/export")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert not any("outside" in n or "secret-link" in n for n in z.namelist())


def test_export_enforces_size_limit(app_client, sample_profile, monkeypatch):
    seed_state(sample_profile)
    monkeypatch.setenv("CLOAK_PROFILE_EXPORT_MAX_BYTES", "8")
    assert app_client.post(f"/api/profiles/{sample_profile['id']}/export").status_code == 413


@pytest.mark.asyncio
async def test_cancelled_export_holds_source_until_worker_finishes(sample_profile, monkeypatch):
    import asyncio
    import threading

    from backend import main, profile_exports
    from backend.browser_manager import ProfileBusyError

    seed_state(sample_profile)
    entered, release = threading.Event(), threading.Event()
    original = profile_exports._build
    built = []

    def delayed(*args):
        entered.set()
        release.wait(5)
        result = original(*args)
        built.append(result)
        return result

    monkeypatch.setattr(profile_exports, "_build", delayed)
    task = asyncio.create_task(main.export_profile(sample_profile["id"]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        with pytest.raises(ProfileBusyError):
            async with main.browser_mgr.hold_stopped(sample_profile["id"]):
                pass
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert built and not built[0].parent.exists()
        async with main.browser_mgr.hold_stopped(sample_profile["id"]):
            pass
    finally:
        release.set()


@pytest.mark.asyncio
async def test_response_disconnect_cleans_archive(sample_profile):
    from backend import main

    seed_state(sample_profile)
    response = await main.export_profile(sample_profile["id"])
    archive = Path(response.path)

    async def disconnected(message):
        raise OSError("client disconnected")

    async def receive():
        return {"type": "http.request"}

    with pytest.raises(OSError):
        await response({"type": "http", "method": "POST", "extensions": {}, "headers": []}, receive, disconnected)
    assert not archive.parent.exists()


def test_successful_export_cleans_archive_after_response(app_client, sample_profile, monkeypatch):
    from backend import profile_exports

    seed_state(sample_profile)
    original = profile_exports._build
    built = []

    def tracked(*args):
        path = original(*args)
        built.append(path)
        return path

    monkeypatch.setattr(profile_exports, "_build", tracked)
    assert app_client.post(f"/api/profiles/{sample_profile['id']}/export").status_code == 200
    assert built and not built[0].parent.exists()


def test_export_requires_auth_config_and_rejects_native_profiles(app_client, sample_profile, monkeypatch):
    from backend import main
    from backend.runtime import RuntimeConfig

    seed_state(sample_profile)
    monkeypatch.setattr(main, "AUTH_TOKEN", None)
    assert app_client.post(f"/api/profiles/{sample_profile['id']}/export").status_code == 503
    monkeypatch.setattr(main, "AUTH_TOKEN", "local-export-test")
    monkeypatch.setattr(
        main.browser_mgr,
        "runtime",
        RuntimeConfig("windows", "native", "native-window", Path(sample_profile["user_data_dir"]).parent),
    )
    assert app_client.post(f"/api/profiles/{sample_profile['id']}/export").status_code == 409


def test_export_preserves_geoip_and_safe_custom_flags(app_client, sample_profile):
    seed_state(sample_profile)
    db.update_profile(
        sample_profile["id"], geoip=True, launch_args=["--fingerprint-platform=macos", "--fingerprint-webrtc-ip=auto"]
    )
    r = app_client.post(f"/api/profiles/{sample_profile['id']}/export")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        m = json.loads(z.read("profile.json"))
        assert m["geoip"]
        assert m["fingerprint_args"][-2:] == ["--fingerprint-platform=macos", "--fingerprint-webrtc-ip=auto"]
    db.update_profile(sample_profile["id"], launch_args=["--proxy-server=secret"])
    assert app_client.post(f"/api/profiles/{sample_profile['id']}/export").status_code == 409


@pytest.mark.parametrize("limit", ["not-an-integer", "0", "-1"])
def test_invalid_export_limit_is_unavailable(app_client, sample_profile, monkeypatch, limit):
    seed_state(sample_profile)
    monkeypatch.setenv("CLOAK_PROFILE_EXPORT_MAX_BYTES", limit)
    assert app_client.post(f"/api/profiles/{sample_profile['id']}/export").status_code == 503


def test_export_aborts_and_cleans_archive_on_scan_error(app_client, sample_profile, monkeypatch, tmp_path):
    from backend import profile_exports

    root = seed_state(sample_profile)
    directory = tmp_path / "failed-export"

    def make_directory(**kwargs):
        directory.mkdir()
        return str(directory)

    def unreadable_walk(path, *, followlinks, onerror=None):
        yield str(root), [], ["Local State"]
        if onerror is not None:
            onerror(PermissionError("browser storage directory is unreadable"))

    monkeypatch.setattr(profile_exports.tempfile, "mkdtemp", make_directory)
    monkeypatch.setattr(profile_exports.os, "walk", unreadable_walk)
    with pytest.raises(PermissionError, match="browser storage directory is unreadable"):
        app_client.post(f"/api/profiles/{sample_profile['id']}/export")
    assert not directory.exists()


@pytest.mark.asyncio
async def test_slow_downloads_bound_archives_until_response_cleanup(sample_profile, monkeypatch):
    import asyncio

    from backend import main, profile_exports

    seed_state(sample_profile)
    monkeypatch.setattr(profile_exports, "_SLOTS", asyncio.Semaphore(2))
    first = await main.export_profile(sample_profile["id"])
    second = await main.export_profile(sample_profile["id"])
    third = asyncio.create_task(main.export_profile(sample_profile["id"]))
    entered, release = asyncio.Event(), asyncio.Event()

    async def receive():
        return {"type": "http.request"}

    async def slow_disconnect(message):
        entered.set()
        await release.wait()
        raise OSError("client disconnected")

    async def consume(message):
        pass

    scope = {"type": "http", "method": "POST", "extensions": {}, "headers": []}
    download = asyncio.create_task(first(scope, receive, slow_disconnect))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.sleep(0.05)
        assert not third.done(), "A third archive must wait while two downloads retain their ZIPs"
        assert Path(first.path).exists() and Path(second.path).exists()
        release.set()
        with pytest.raises(OSError, match="client disconnected"):
            await download
        assert not Path(first.path).parent.exists()
        last = await asyncio.wait_for(third, 2)
        await second(scope, receive, consume)
        await last(scope, receive, consume)
        assert not Path(second.path).parent.exists() and not Path(last.path).parent.exists()
    finally:
        release.set()
        if not download.done():
            await asyncio.gather(download, return_exceptions=True)
        if not third.done():
            third.cancel()
        results = await asyncio.gather(third, return_exceptions=True)
        if Path(second.path).exists():
            await second(scope, receive, consume)
        if not isinstance(results[0], BaseException) and Path(results[0].path).exists():
            await results[0](scope, receive, consume)


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_slot_until_archive_removed(sample_profile, monkeypatch):
    import asyncio
    import threading

    from backend import main, profile_exports

    seed_state(sample_profile)
    slots = asyncio.Semaphore(1)
    monkeypatch.setattr(profile_exports, "_SLOTS", slots)
    response = await main.export_profile(sample_profile["id"])
    entered, release = threading.Event(), threading.Event()
    original = profile_exports.shutil.rmtree

    def delayed_cleanup(*args):
        entered.set()
        release.wait(5)
        return original(*args)

    monkeypatch.setattr(profile_exports.shutil, "rmtree", delayed_cleanup)

    async def receive():
        return {"type": "http.request"}

    async def consume(message):
        pass

    scope = {"type": "http", "method": "POST", "extensions": {}, "headers": []}
    task = asyncio.create_task(response(scope, receive, consume))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert slots.locked() and Path(response.path).exists()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not slots.locked() and not Path(response.path).parent.exists()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_build_releases_archive_slot(sample_profile, monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from backend import main, profile_exports

    seed_state(sample_profile)
    slots = asyncio.Semaphore(1)
    monkeypatch.setattr(profile_exports, "_SLOTS", slots)
    monkeypatch.setenv("CLOAK_PROFILE_EXPORT_MAX_BYTES", "8")
    with pytest.raises(HTTPException) as error:
        await main.export_profile(sample_profile["id"])
    assert error.value.status_code == 413
    assert not slots.locked()
