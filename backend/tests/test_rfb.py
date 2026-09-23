"""Tests for RFB binary protocol parsing functions in main.py."""

from __future__ import annotations

import struct

from backend.main import (
    _build_server_cut_text,
    _filter_rfb_client_messages,
    _parse_kasmvnc_clipboard,
    _rewrite_pointer_event,
    _rewrite_set_encodings,
    _rfb_msg_length,
    _ALLOWED_ENCODINGS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_key_event(down: int = 1, key: int = 0x61) -> bytes:
    """Build an 8-byte RFB KeyEvent (type 4)."""
    return struct.pack(">BBxxI", 4, down, key)


def _make_pointer_event(mask: int = 0, x: int = 100, y: int = 200) -> bytes:
    """Build a 6-byte RFB PointerEvent (type 5)."""
    return struct.pack(">BBHH", 5, mask, x, y)


def _make_fb_update_request(x: int = 0, y: int = 0, w: int = 1920, h: int = 1080, incr: int = 1) -> bytes:
    """Build a 10-byte FramebufferUpdateRequest (type 3)."""
    return struct.pack(">BBHHHH", 3, incr, x, y, w, h)


def _make_set_encodings(encodings: list[int]) -> bytes:
    """Build a SetEncodings message (type 2)."""
    header = struct.pack(">BxH", 2, len(encodings))
    for enc in encodings:
        header += struct.pack(">i", enc)  # signed
    return header


def _make_client_cut_text(text: str) -> bytes:
    """Build a ClientCutText message (type 6)."""
    text_bytes = text.encode("latin-1", errors="replace")
    return struct.pack(">BxxxI", 6, len(text_bytes)) + text_bytes


def _make_extension_150() -> bytes:
    """Build a 10-byte EnableContinuousUpdates extension (type 150)."""
    return struct.pack(">BBHHHH", 150, 1, 0, 0, 1920, 1080)


def _make_kasmvnc_clipboard(*entries: tuple[str, bytes]) -> bytes:
    """Build a KasmVNC BinaryClipboard message (type 180) in its real wire format:
    type(1) count(1), then per entry id(4) mime_len(1) mime data_len(4) data."""
    buf = bytearray([180, len(entries)])
    for i, (mime, data) in enumerate(entries):
        buf += struct.pack(">IB", i, len(mime)) + mime.encode() + struct.pack(">I", len(data)) + data
    return bytes(buf)


# Captured from KasmVNC 1.3.3: Chrome copying "from-input" out of an <input>.
_REAL_CLIPBOARD_INPUT = bytes.fromhex("b401000000000a746578742f706c61696e0000000a66726f6d2d696e707574")


# ── _parse_kasmvnc_clipboard ────────────────────────────────────────────────


def test_parse_clipboard_text_plain():
    data = _make_kasmvnc_clipboard(("text/plain", b"hello"))
    assert _parse_kasmvnc_clipboard(data) == "hello"


def test_parse_clipboard_real_capture():
    assert _parse_kasmvnc_clipboard(_REAL_CLIPBOARD_INPUT) == "from-input"


def test_parse_clipboard_no_text_plain():
    data = _make_kasmvnc_clipboard(("image/png", b"binary"))
    assert _parse_kasmvnc_clipboard(data) is None


def test_parse_clipboard_too_short():
    assert _parse_kasmvnc_clipboard(b"\xb4\x01\x00") is None


def test_parse_clipboard_empty_text():
    data = _make_kasmvnc_clipboard(("text/plain", b""))
    assert _parse_kasmvnc_clipboard(data) == ""


def test_parse_clipboard_multiple_mimes():
    """text/plain after another entry is found (each entry carries its own id)."""
    data = _make_kasmvnc_clipboard(("text/html", b"<b>world</b>"), ("text/plain", b"world"))
    assert _parse_kasmvnc_clipboard(data) == "world"


def test_parse_clipboard_truncated_text_is_rejected():
    data = _make_kasmvnc_clipboard(("text/plain", b"complete text"))
    assert _parse_kasmvnc_clipboard(data[:-3]) is None


def test_parse_clipboard_utf8():
    data = _make_kasmvnc_clipboard(("text/plain", "café".encode()))
    assert _parse_kasmvnc_clipboard(data) == "café"


# ── _build_server_cut_text ───────────────────────────────────────────────────


def test_build_cut_text_basic():
    result = _build_server_cut_text("hi")
    assert result[0] == 3  # ServerCutText type
    length = struct.unpack_from(">I", result, 4)[0]
    assert length == 2
    assert result[8:] == b"hi"


def test_build_cut_text_empty():
    result = _build_server_cut_text("")
    assert result[0] == 3
    length = struct.unpack_from(">I", result, 4)[0]
    assert length == 0
    assert len(result) == 8


def test_build_cut_text_latin1_fallback():
    # CJK chars are outside Latin-1 — replaced with '?'
    result = _build_server_cut_text("hello \u65e5\u672c")
    text_bytes = result[8:]
    assert b"?" in text_bytes  # unicode replaced
    assert text_bytes.startswith(b"hello ")


# ── _rfb_msg_length ─────────────────────────────────────────────────────────


def test_rfb_len_set_pixel_format():
    data = bytes(20)  # type 0, 20 bytes
    assert _rfb_msg_length(data, 0) == 20


def test_rfb_len_fb_update_request():
    data = _make_fb_update_request()
    assert _rfb_msg_length(data, 0) == 10


def test_rfb_len_key_event():
    data = _make_key_event()
    assert _rfb_msg_length(data, 0) == 8


def test_rfb_len_pointer_event():
    data = _make_pointer_event()
    assert _rfb_msg_length(data, 0) == 6


def test_rfb_len_set_encodings():
    data = _make_set_encodings([0, 1, 2])
    assert _rfb_msg_length(data, 0) == 4 + 3 * 4  # 16


def test_rfb_len_client_cut_text():
    data = _make_client_cut_text("hello world")
    assert _rfb_msg_length(data, 0) == 8 + 11  # 19


def test_rfb_len_extension_150():
    data = _make_extension_150()
    assert _rfb_msg_length(data, 0) == 10


def test_rfb_len_unknown():
    data = bytes([99])  # unknown type
    assert _rfb_msg_length(data, 0) is None


def test_rfb_len_with_offset():
    """Length calculation works when message is not at start of buffer."""
    prefix = b"\x00" * 10
    key = _make_key_event()
    data = prefix + key
    assert _rfb_msg_length(data, 10) == 8


# ── _rewrite_set_encodings ──────────────────────────────────────────────────


def test_rewrite_keeps_allowed():
    allowed = [0, 1, 2, 5, 7]
    data = _make_set_encodings(allowed)
    result = _rewrite_set_encodings(data, 0, len(data))
    # All kept — output should be identical
    assert result == data


def test_rewrite_strips_disallowed():
    # -260 and -307 are not in _ALLOWED_ENCODINGS
    data = _make_set_encodings([0, 1, -260, -307])
    result = _rewrite_set_encodings(data, 0, len(data))
    # Only 0 and 1 should remain
    num_enc = struct.unpack_from(">H", result, 2)[0]
    assert num_enc == 2
    enc1 = struct.unpack_from(">i", result, 4)[0]
    enc2 = struct.unpack_from(">i", result, 8)[0]
    assert enc1 == 0
    assert enc2 == 1


def test_rewrite_with_offset():
    """Rewrite works when message is at a non-zero offset."""
    prefix = b"\xff" * 8
    data = _make_set_encodings([0, -260])
    full = prefix + data
    result = _rewrite_set_encodings(full, 8, len(data))
    num_enc = struct.unpack_from(">H", result, 2)[0]
    assert num_enc == 1  # only 0 kept


# ── _rewrite_pointer_event ───────────────────────────────────────────────────


def test_rewrite_pointer_basic():
    data = _make_pointer_event(mask=1, x=100, y=200)
    result = _rewrite_pointer_event(data, 0)
    assert len(result) == 11  # KasmVNC format
    assert result[0] == 5  # PointerEvent type preserved
    mask_u16 = struct.unpack_from(">H", result, 1)[0]
    x = struct.unpack_from(">H", result, 3)[0]
    y = struct.unpack_from(">H", result, 5)[0]
    sx = struct.unpack_from(">h", result, 7)[0]
    sy = struct.unpack_from(">h", result, 9)[0]
    assert mask_u16 == 1
    assert x == 100
    assert y == 200
    assert sx == 0
    assert sy == 0


def test_rewrite_pointer_mask_expansion():
    """u8 mask 0xFF should be expanded to u16 0x00FF."""
    data = _make_pointer_event(mask=0xFF, x=0, y=0)
    result = _rewrite_pointer_event(data, 0)
    mask_u16 = struct.unpack_from(">H", result, 1)[0]
    assert mask_u16 == 0x00FF


def test_rewrite_pointer_with_offset():
    prefix = b"\x00" * 4
    data = _make_pointer_event(mask=2, x=50, y=75)
    full = prefix + data
    result = _rewrite_pointer_event(full, 4)
    assert len(result) == 11
    mask_u16 = struct.unpack_from(">H", result, 1)[0]
    assert mask_u16 == 2


# ── _filter_rfb_client_messages ──────────────────────────────────────────────


def test_filter_keeps_standard_types():
    key = _make_key_event()
    fb = _make_fb_update_request()
    data = key + fb
    result = _filter_rfb_client_messages(data)
    # KeyEvent stays 8 bytes, FBUpdateRequest stays 10 bytes
    # Total = 18
    assert len(result) == 18
    assert result[:8] == key
    assert result[8:] == fb


def test_filter_forwards_enable_continuous_updates():
    key1 = _make_key_event(down=1, key=0x61)
    ext = _make_extension_150()
    key2 = _make_key_event(down=0, key=0x61)
    data = key1 + ext + key2
    assert _filter_rfb_client_messages(data) == data


def _make_client_fence(payload: bytes = b"\x01\x02\x03\x04") -> bytes:
    return struct.pack(">BxxxIB", 248, 0x00000001, len(payload)) + payload


def test_rfb_len_client_fence():
    assert _rfb_msg_length(_make_client_fence(), 0) == 13
    assert _rfb_msg_length(_make_client_fence(b""), 0) == 9


def test_filter_forwards_client_fence_between_messages():
    """noVNC's fence reply is variable length; the next message must still be found."""
    key = _make_key_event()
    fence = _make_client_fence(b"abcdefgh")
    req = _make_fb_update_request()
    data = key + fence + req
    assert _filter_rfb_client_messages(data) == data


def test_rewrite_keeps_fence_and_continuous_updates():
    data = _make_set_encodings([7, -312, -313])
    assert _rewrite_set_encodings(data, 0, len(data)) == data


def test_filter_reports_client_pixel_format():
    pf = struct.pack(">BBBBHHHBBBxxx", 16, 16, 0, 1, 31, 63, 31, 11, 5, 0)
    seen = []
    data = b"\x00\x00\x00\x00" + pf + _make_key_event()
    assert _filter_rfb_client_messages(data, seen.append) == data
    assert seen == [pf]


def test_filter_drops_unknown():
    key = _make_key_event()
    unknown = bytes([99, 0, 0, 0, 0])  # unknown type with some trailing bytes
    data = key + unknown
    result = _filter_rfb_client_messages(data)
    # Key event kept, unknown causes break (rest dropped)
    assert result == key


def test_filter_drops_incomplete():
    # KeyEvent is 8 bytes, give only 4
    data = _make_key_event()[:4]
    result = _filter_rfb_client_messages(data)
    assert result == b""


def test_filter_rewrites_pointer():
    """PointerEvent should be rewritten from 6→11 bytes."""
    ptr = _make_pointer_event(mask=1, x=100, y=200)
    result = _filter_rfb_client_messages(ptr)
    assert len(result) == 11  # expanded
    assert result[0] == 5


def test_filter_rewrites_set_encodings():
    """SetEncodings should have disallowed encodings stripped."""
    enc = _make_set_encodings([0, 1, -260])
    result = _filter_rfb_client_messages(enc)
    num_enc = struct.unpack_from(">H", result, 2)[0]
    assert num_enc == 2  # -260 stripped


def test_filter_mixed_frame():
    """Realistic frame: KeyEvent + unsupported xvp + PointerEvent + ClientCutText."""
    key = _make_key_event()        # 8 bytes, kept
    xvp = bytes([252, 0, 1, 2])    # 4 bytes, stripped
    ptr = _make_pointer_event()    # 6 bytes → 11 bytes (rewritten)
    cut = _make_client_cut_text("hi")  # 8+2=10 bytes, kept

    data = key + xvp + ptr + cut
    result = _filter_rfb_client_messages(data)

    # key(8) + ptr_rewritten(11) + cut(10) = 29
    assert len(result) == 29
    assert result[0] == 4   # KeyEvent type
    assert result[8] == 5   # PointerEvent type (rewritten)
    assert result[19] == 6  # ClientCutText type


def test_filter_empty_input():
    assert _filter_rfb_client_messages(b"") == b""
