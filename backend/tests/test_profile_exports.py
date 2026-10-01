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
