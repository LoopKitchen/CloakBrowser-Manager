"""Tests for the KasmVNC → client stream translator (backend/rfb_stream.py)."""

from __future__ import annotations

import random
import struct

import pytest

from backend import rfb_stream
from backend.rfb_stream import ServerStreamTranslator, build_server_cut_text

# ── Stream builders ──────────────────────────────────────────────────────────


def _pf(bpp=32, depth=24, true_colour=1, rmax=255, gmax=255, bmax=255, rs=16, gs=8, bs=0) -> bytes:
    return struct.pack(">BBBBHHHBBBxxx", bpp, depth, 0, true_colour, rmax, gmax, bmax, rs, gs, bs)


def _handshake(pf: bytes = _pf(), w=1920, h=1080, name=b"kasm") -> bytes:
    server_init = struct.pack(">HH", w, h) + pf + struct.pack(">I", len(name)) + name
    return b"RFB 003.008\n" + b"\x01\x01" + b"\x00\x00\x00\x00" + server_init


def _fbu(*rects: bytes, last_rect: bool = False) -> bytes:
    if last_rect:
        return struct.pack(">BxH", 0, 0xFFFF) + b"".join(rects) + _rect(0, 0, 0, 0, -224)
    return struct.pack(">BxH", 0, len(rects)) + b"".join(rects)


def _rect(x, y, w, h, enc, payload=b"") -> bytes:
    return struct.pack(">HHHHi", x, y, w, h, enc) + payload


def _compact(n: int) -> bytes:
    out = bytearray([n & 0x7F])
    if n > 0x7F:
        out[0] |= 0x80
        out.append((n >> 7) & 0x7F)
        if n > 0x3FFF:
            out[1] |= 0x80
            out.append((n >> 14) & 0xFF)
    return bytes(out)


def _clipboard(*entries: tuple[str, bytes]) -> bytes:
    buf = bytearray([180, len(entries)])
    for i, (mime, data) in enumerate(entries):
        buf += struct.pack(">IB", i, len(mime)) + mime.encode() + struct.pack(">I", len(data)) + data
    return bytes(buf)


def _fence(payload=b"\x00\x00\x00\x01") -> bytes:
    return struct.pack(">BxxxIB", 248, 0x80000003, len(payload)) + payload


def _hextile_payload(w: int, h: int, px: int = 4) -> bytes:
    """One tile of every shape: raw, bg+fg+plain subrects, coloured subrects, bg only."""
    shapes = [
        lambda tw, th: b"\x01" + b"\x11" * (tw * th * px),
        lambda tw, th: b"\x0e" + b"\x22" * px + b"\x33" * px + b"\x02" + b"\x00\x11" * 2,
        lambda tw, th: b"\x18" + b"\x03" + (b"\x44" * px + b"\x00\x11") * 3,
        lambda tw, th: b"\x02" + b"\x55" * px,
    ]
    out, i = bytearray(), 0
    for ty in range(0, h, 16):
        for tx in range(0, w, 16):
            out += shapes[i % len(shapes)](min(16, w - tx), min(16, h - ty))
            i += 1
    return bytes(out)


def _tight_rects(tpixel: int = 3) -> list[bytes]:
    return [
        _rect(0, 0, 64, 64, 7, b"\x80" + b"\x10" * tpixel),                                 # fill
        _rect(0, 0, 64, 64, 7, b"\x90" + _compact(5) + b"JPEG!"),                          # jpeg, 1-byte length
        _rect(0, 0, 64, 64, 7, b"\x90" + _compact(300) + b"j" * 300),                       # jpeg, 2-byte length
        _rect(0, 0, 64, 64, 7, b"\x90" + _compact(20000) + b"J" * 20000),                   # jpeg, 3-byte length
        _rect(0, 0, 64, 64, 7, b"\xb0" + _compact(7) + b"webp..."),                         # webp
        _rect(0, 0, 64, 64, 7, b"\xc0" + _compact(6) + b"qoi..."),                          # qoi
        _rect(0, 0, 2, 1, 7, b"\x00" + b"\x01" * (2 * tpixel)),                             # copy, raw (<12 bytes)
        _rect(0, 0, 8, 8, 7, b"\x00" + _compact(9) + b"z" * 9),                             # copy, zlib
        _rect(0, 0, 12 // tpixel, 1, 7, b"\x00" + _compact(5) + b"c" * 5),                 # exactly 12 bytes: zlib
        _rect(0, 0, 10, 4, 7, b"\x40\x01\x01" + b"\x07" * (2 * tpixel) + b"\xff" * 8),     # 2-colour palette, raw bits
        _rect(0, 0, 30, 30, 7, b"\x50\x01\x04" + b"\x07" * (5 * tpixel) + _compact(11) + b"p" * 11),  # 5 colours
        _rect(0, 0, 16, 16, 7, b"\x60\x02" + _compact(12) + b"g" * 12),                     # gradient
    ]


def _all_messages() -> tuple[bytes, bytes]:
    """A stream covering every message and encoding, and what the client must receive."""
    clip = _clipboard(("text/html", b"<b>copied</b>"), ("text/plain", b"copied"))
    parts: list[tuple[bytes, bytes]] = [
        (_handshake(), None),
        (_fbu(_rect(0, 0, 3, 2, 0, b"\xab" * 24), _rect(5, 5, 10, 10, 1, b"\x00\x01\x00\x02")), None),
        (clip, build_server_cut_text("copied")),
        (_fbu(*_tight_rects()), None),
        (_fence(), None),
        (b"\x96", None),  # EndOfContinuousUpdates
        (_fbu(_rect(0, 0, 40, 20, 5, _hextile_payload(40, 20))), None),
        (_fbu(_rect(0, 0, 8, 8, 2, struct.pack(">I", 2) + b"\x01" * 4 + (b"\x02" * 4 + b"\x00" * 8) * 2)), None),
        (_fbu(_rect(0, 0, 64, 64, 16, struct.pack(">I", 6) + b"zrle!!")), None),
        (_fbu(_rect(0, 0, 9, 3, -239, b"\x05" * (9 * 3 * 4) + b"\x06" * (2 * 3))), None),
        (_fbu(_rect(0, 0, 3, 3, 0, b"\x01" * 36), _rect(0, 0, 64, 64, 7, b"\x80" + b"\x00" * 3), last_rect=True), None),
        (b"\x02", None),  # Bell
        (struct.pack(">BxxxI", 3, 5) + b"hello", None),   # ServerCutText
        (struct.pack(">BxxxI", 178, 4) + b"stat", None),  # Stats
        (struct.pack(">BxHH", 1, 0, 2) + b"\x00" * 12, None),  # SetColourMapEntries
        (_fbu(_rect(0, 0, 1280, 720, -223)), None),     # DesktopSize
        (_clipboard(("image/png", b"\x89PNG")), b""),     # nothing textual: dropped
        (_clipboard(("text/plain", b"")), b""),           # empty: dropped, never clears the viewer
        (_fbu(_rect(0, 0, 1, 1, 0, b"\x09" * 4)), None),
    ]
    return b"".join(i for i, _ in parts), b"".join(i if o is None else o for i, o in parts)


def _feed_chunks(data: bytes, cuts: list[int]) -> tuple[bytes, ServerStreamTranslator]:
    t = ServerStreamTranslator()
    out, prev = bytearray(), 0
    for cut in [*cuts, len(data)]:
        out += t.feed(data[prev:cut])
        prev = cut
    return bytes(out), t


# ── Behaviour ────────────────────────────────────────────────────────────────


def test_translates_whole_stream():
    data, expected = _all_messages()
    out, t = _feed_chunks(data, [])
    assert out == expected
    assert not t.passthrough
    assert t.clipboards == 1


def test_output_is_independent_of_frame_boundaries():
    """KasmVNC may split or join messages across WebSocket frames anywhere."""
    data, expected = _all_messages()
    for cut in range(1, len(data), 7):
        out, t = _feed_chunks(data, [cut])
        assert out == expected, f"split at {cut}"
        assert not t.passthrough


def test_output_is_independent_of_random_chunking():
    data, expected = _all_messages()
    rng = random.Random(1234)
    for _ in range(50):
        cuts = sorted(rng.sample(range(1, len(data)), 40))
        out, t = _feed_chunks(data, cuts)
        assert out == expected
        assert not t.passthrough


def test_byte_by_byte():
    data, expected = _all_messages()
    out, t = _feed_chunks(data, list(range(1, len(data))))
    assert out == expected
    assert not t.passthrough


def test_clipboard_packed_inside_a_frame_is_converted():
    """The production failure: screen data and a clipboard announcement in one frame."""
    t = ServerStreamTranslator()
    t.feed(_handshake())
    frame = _fbu(_rect(0, 0, 64, 64, 7, b"\x90" + _compact(4) + b"jpeg")) + _clipboard(("text/plain", b"stress-07"))
    out = t.feed(frame + _fence())
    cut = build_server_cut_text("stress-07")
    assert cut in out
    assert b"\xb4" not in out
    assert out.endswith(_fence())


def test_real_kasmvnc_clipboard_after_screen_data():
    real = bytes.fromhex("b401000000000a746578742f706c61696e0000000a66726f6d2d696e707574")
    t = ServerStreamTranslator()
    t.feed(_handshake())
    out = t.feed(_fbu(_rect(0, 0, 1, 1, 0, b"\x00" * 4)) + real)
    assert out.endswith(build_server_cut_text("from-input"))


def test_clipboard_split_across_frames_is_held_until_complete():
    t = ServerStreamTranslator()
    t.feed(_handshake())
    clip = _clipboard(("text/plain", b"x" * 1000))
    assert t.feed(clip[:500]) == b""
    assert t.feed(clip[500:]) == build_server_cut_text("x" * 1000)


def test_screen_data_is_forwarded_before_its_rect_completes():
    """Payload bytes are not held back, so large rects add no latency."""
    t = ServerStreamTranslator()
    t.feed(_handshake())
    head = _fbu(_rect(0, 0, 100, 100, 0))  # header only; 40 000 payload bytes follow
    assert t.feed(head) == head
    assert t.feed(b"\x01" * 1000) == b"\x01" * 1000


def test_client_pixel_format_changes_payload_sizes():
    t = ServerStreamTranslator()
    t.feed(_handshake())
    t.set_pixel_format(_pf(bpp=16, depth=16, rmax=31, gmax=63, bmax=31, rs=11, gs=5, bs=0))
    stream = _fbu(_rect(0, 0, 3, 3, 0, b"\x01" * 18), *_tight_rects(tpixel=2)) + _clipboard(("text/plain", b"ok"))
    assert t.feed(stream).endswith(build_server_cut_text("ok"))
    assert not t.passthrough


def test_unknown_message_falls_back_to_passthrough():
    t = ServerStreamTranslator()
    t.feed(_handshake())
    odd = bytes([77, 1, 2, 3])
    assert t.feed(odd) == odd
    assert t.passthrough
    # Passthrough keeps the previous behaviour: forward, convert 180 only at a frame start.
    assert t.feed(b"\x00\x01\x02") == b"\x00\x01\x02"
    assert t.feed(_clipboard(("text/plain", b"hi"))) == build_server_cut_text("hi")


def test_unknown_rect_encoding_falls_back_to_passthrough():
    t = ServerStreamTranslator()
    t.feed(_handshake())
    frame = _fbu(_rect(0, 0, 4, 4, 99, b"????"))
    assert t.feed(frame) == frame
    assert t.passthrough


def test_non_none_security_falls_back_to_passthrough():
    t = ServerStreamTranslator()
    data = b"RFB 003.008\n" + b"\x01\x02" + b"\x00" * 16
    assert t.feed(data) == data
    assert t.passthrough


def test_oversized_clipboard_text_is_discarded_not_forwarded(monkeypatch):
    """Type 180 must never reach noVNC, however large the clipboard."""
    monkeypatch.setattr(rfb_stream, "MAX_CLIPBOARD_TEXT", 100)
    t = ServerStreamTranslator()
    t.feed(_handshake())
    data = _clipboard(("text/plain", b"x" * 1000)) + b"\x02"
    out = b"".join(t.feed(data[i:i + 64]) for i in range(0, len(data), 64))
    assert out == b"\x02"
    assert not t.passthrough
    assert t.clipboards == 0


def test_non_text_clipboard_entries_are_streamed_not_buffered():
    t = ServerStreamTranslator()
    t.feed(_handshake())
    clip = _clipboard(("image/png", b"\x89" * 100_000), ("text/plain", b"after image"))
    out, peak = bytearray(), 0
    for i in range(0, len(clip), 4096):
        out += t.feed(clip[i:i + 4096])
        peak = max(peak, len(t._buf))
    assert bytes(out) == build_server_cut_text("after image")
    assert peak <= 4096


def test_first_text_entry_wins_and_later_entries_are_discarded():
    t = ServerStreamTranslator()
    t.feed(_handshake())
    clip = _clipboard(("text/plain", b"first"), ("text/plain", b"second"), ("text/html", b"<i>x</i>"))
    assert t.feed(clip + b"\x02") == build_server_cut_text("first") + b"\x02"


def test_passthrough_applies_the_same_clipboard_text_limit(monkeypatch):
    monkeypatch.setattr(rfb_stream, "MAX_CLIPBOARD_TEXT", 8)
    t = ServerStreamTranslator()
    t.feed(_handshake())
    t.feed(bytes([77]))  # unknown message: passthrough from here
    assert t.passthrough
    assert t.feed(_clipboard(("text/plain", b"0123456789"))) == b""
    assert t.feed(_clipboard(("text/plain", b"short"))) == build_server_cut_text("short")


def test_unit_that_never_completes_falls_back(monkeypatch):
    monkeypatch.setattr(rfb_stream, "MAX_HELD_MESSAGE", 5)
    t = ServerStreamTranslator()
    t.feed(_handshake())
    partial = _fbu(_rect(0, 0, 1, 1, 0, b"\x00" * 4))[:12]  # update header + 8 of 12 rect-header bytes
    assert t.feed(partial) == partial
    assert t.passthrough


def test_unaligned_888_format_uses_full_width_tight_pixels():
    """KasmVNC's is888() also requires byte-aligned shifts; otherwise Tight sends 4 bytes."""
    t = ServerStreamTranslator()
    t.feed(_handshake())
    t.set_pixel_format(_pf(rs=17, gs=9, bs=1))
    stream = _fbu(_rect(0, 0, 64, 64, 7, b"\x80" + b"\x10" * 4)) + _clipboard(("text/plain", b"ok"))
    assert t.feed(stream).endswith(build_server_cut_text("ok"))
    assert not t.passthrough


def test_aligned_888_format_with_other_shifts_uses_three_byte_pixels():
    t = ServerStreamTranslator()
    t.feed(_handshake())
    t.set_pixel_format(_pf(rs=0, gs=8, bs=16))  # BGR order, still byte aligned
    stream = _fbu(_rect(0, 0, 64, 64, 7, b"\x80" + b"\x10" * 3)) + _clipboard(("text/plain", b"ok"))
    assert t.feed(stream).endswith(build_server_cut_text("ok"))
    assert not t.passthrough


@pytest.mark.parametrize("n", [0, 1, 0x7F, 0x80, 0x3FFF, 0x4000, 0x3FFFFF])
def test_compact_length_round_trip(n):
    t = ServerStreamTranslator()
    t.feed(_handshake())
    frame = _fbu(_rect(0, 0, 64, 64, 7, b"\x90" + _compact(n) + b"j" * n)) + b"\x02"
    assert t.feed(frame) == frame
    assert not t.passthrough
