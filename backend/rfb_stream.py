"""KasmVNC → noVNC server-stream translation.

KasmVNC announces clipboard changes with its own BinaryClipboard message (type 180),
which noVNC does not understand. The proxy rewrites it into a standard ServerCutText
(type 3). KasmVNC freely packs that message into the same WebSocket frame as screen
data, so it cannot be found by looking at frame starts: the stream has to be walked
message by message. ``ServerStreamTranslator`` does that with a length-only parser —
it never decodes pixels, it only needs to know how many bytes each message occupies.

Anything the parser does not recognise switches the connection to passthrough, which
forwards bytes untouched and converts type 180 only at a frame start (the pre-parser
behaviour), so an unknown message can never make things worse than before.
"""

from __future__ import annotations

import logging
import struct
from typing import Callable, Optional

logger = logging.getLogger("cloakbrowser.manager")

# Largest single message held back for conversion (a clipboard carrying an image).
MAX_HELD_MESSAGE = 32 * 1024 * 1024

_BINARY_CLIPBOARD = 180


def parse_kasmvnc_clipboard(data: bytes) -> Optional[str]:
    """Return the text/plain entry of a KasmVNC BinaryClipboard message, or None.

    Wire format (KasmVNC 1.3 SMsgWriter::writeBinaryClipboard):
    type(1) count(1), then per entry: id(4) mime_len(1) mime data_len(4) data.
    """
    if len(data) < 2 or data[0] != _BINARY_CLIPBOARD:
        return None
    count, off = data[1], 2
    for _ in range(count):
        if off + 5 > len(data):
            return None
        mime_len = data[off + 4]
        off += 5
        if off + mime_len + 4 > len(data):
            return None
        mime = bytes(data[off:off + mime_len])
        off += mime_len
        data_len = struct.unpack_from(">I", data, off)[0]
        off += 4
        if mime.split(b";", 1)[0].strip() == b"text/plain":
            return bytes(data[off:off + data_len]).decode("utf-8", errors="replace")
        off += data_len
    return None


def build_server_cut_text(text: str) -> bytes:
    """Standard RFB ServerCutText (type 3). The spec mandates Latin-1; others become '?'."""
    text_bytes = text.encode("latin-1", errors="replace")
    return struct.pack(">BxxxI", 3, len(text_bytes)) + text_bytes


class _NeedMore(Exception):
    """The buffer ends before the unit being parsed does."""


class _Desync(Exception):
    """The stream holds something the parser does not understand."""


class _Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytearray) -> None:
        self.buf, self.pos = buf, 0

    def need(self, n: int) -> None:
        if self.pos + n > len(self.buf):
            raise _NeedMore

    def u8(self) -> int:
        self.need(1)
        self.pos += 1
        return self.buf[self.pos - 1]

    def u16(self) -> int:
        self.need(2)
        self.pos += 2
        return struct.unpack_from(">H", self.buf, self.pos - 2)[0]

    def u32(self) -> int:
        self.need(4)
        self.pos += 4
        return struct.unpack_from(">I", self.buf, self.pos - 4)[0]

    def s32(self) -> int:
        self.need(4)
        self.pos += 4
        return struct.unpack_from(">i", self.buf, self.pos - 4)[0]

    def skip(self, n: int) -> None:
        self.need(n)
        self.pos += n

    def compact_len(self) -> int:
        """Tight's 1-3 byte length."""
        b = self.u8()
        value = b & 0x7F
        if b & 0x80:
            b = self.u8()
            value |= (b & 0x7F) << 7
            if b & 0x80:
                value |= self.u8() << 14
        return value


# A unit is (header bytes consumed, replacement bytes or None to forward them as-is,
# payload bytes that follow and are forwarded untouched).
_Unit = tuple[int, Optional[bytes], int]


class ServerStreamTranslator:
    """Walks the KasmVNC → client RFB stream and rewrites BinaryClipboard in place."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.passthrough = False
        self.clipboards = 0
        self._buf = bytearray()
        self._skip = 0
        self._state: Callable[[_Reader], _Unit] = self._version
        self._rects_left: Optional[int] = 0  # None: until LastRect
        self._hextile: Optional[list[int]] = None  # [x, y, w, h, next_tx, next_ty]
        self._bpp = 4
        self._tpixel = 3

    # -- public ---------------------------------------------------------------

    def set_pixel_format(self, pf: bytes) -> None:
        """Apply a 16-byte RFB PIXEL_FORMAT (ServerInit or the client's SetPixelFormat)."""
        bpp, depth, _big, true_colour, rmax, gmax, bmax = struct.unpack_from(">BBBBHHH", pf, 0)
        self._bpp = max(1, bpp // 8)
        is888 = bpp == 32 and depth == 24 and true_colour and rmax == gmax == bmax == 255
        self._tpixel = 3 if is888 else self._bpp

    def feed(self, data: bytes) -> bytes:
        """Consume one server frame; return the bytes to send to the client (may be empty)."""
        if self.passthrough:
            return self._legacy(data)
        self._buf += data
        out = bytearray()
        try:
            self._run(out)
        except _Desync as exc:
            logger.warning("VNC proxy %s: stream parser lost sync (%s); passthrough from here", self.label, exc)
            self.passthrough = True
            out += self._buf
            self._buf.clear()
            self._skip = 0
        return bytes(out)

    # -- engine ---------------------------------------------------------------

    def _run(self, out: bytearray) -> None:
        while True:
            if self._skip:
                n = min(self._skip, len(self._buf))
                if not n:
                    return
                out += self._buf[:n]
                del self._buf[:n]
                self._skip -= n
                continue
            if not self._buf:
                return
            reader = _Reader(self._buf)
            try:
                consumed, replacement, skip = self._state(reader)
            except _NeedMore:
                if len(self._buf) > MAX_HELD_MESSAGE:
                    raise _Desync(f"message larger than {MAX_HELD_MESSAGE} bytes")
                return
            out += self._buf[:consumed] if replacement is None else replacement
            del self._buf[:consumed]
            self._skip = skip

    def _legacy(self, data: bytes) -> bytes:
        if data and data[0] == _BINARY_CLIPBOARD:
            return self._convert_clipboard(data)
        return data

    def _convert_clipboard(self, msg: bytes) -> bytes:
        text = parse_kasmvnc_clipboard(msg)
        if not text:  # no text/plain entry, or empty: never clear the viewer's clipboard
            logger.info("VNC proxy %s: dropped BinaryClipboard without text", self.label)
            return b""
        self.clipboards += 1
        logger.info("VNC proxy %s: clipboard %d chars", self.label, len(text))
        return build_server_cut_text(text)

    # -- handshake ------------------------------------------------------------

    def _version(self, r: _Reader) -> _Unit:
        r.skip(12)
        if bytes(r.buf[:12]) != b"RFB 003.008\n":
            raise _Desync(f"protocol version {bytes(r.buf[:12])!r}")
        self._state = self._security_types
        return 12, None, 0

    def _security_types(self, r: _Reader) -> _Unit:
        n = r.u8()
        if n == 0:
            raise _Desync("server refused the connection")
        r.skip(n)
        # The Manager runs Xvnc with -SecurityTypes None; any other type carries
        # a handshake of its own that this parser does not follow.
        if bytes(r.buf[1:1 + n]) != b"\x01":
            raise _Desync(f"security types {list(r.buf[1:1 + n])}")
        self._state = self._security_result
        return r.pos, None, 0

    def _security_result(self, r: _Reader) -> _Unit:
        if r.u32() != 0:
            raise _Desync("security handshake failed")
        self._state = self._server_init
        return 4, None, 0

    def _server_init(self, r: _Reader) -> _Unit:
        r.skip(24)
        name_len = struct.unpack_from(">I", r.buf, 20)[0]
        r.skip(name_len)
        self.set_pixel_format(bytes(r.buf[4:20]))
        self._state = self._message
        return r.pos, None, 0

    # -- server messages ------------------------------------------------------

    def _message(self, r: _Reader) -> _Unit:
        t = r.u8()
        if t == 0:  # FramebufferUpdate
            r.skip(1)
            n = r.u16()
            self._rects_left = None if n == 0xFFFF else n
            self._state = self._rect if n else self._message
            return 4, None, 0
        if t == 1:  # SetColourMapEntries
            r.skip(3)
            return 6, None, r.u16() * 6
        if t in (2, 150, 179):  # Bell, EndOfContinuousUpdates, RequestFrameStats
            return 1, None, 0
        if t == 3:  # ServerCutText (a negative length is the extended-clipboard form)
            r.skip(3)
            return 8, None, abs(r.s32())
        if t == 178:  # Stats
            r.skip(3)
            return 8, None, r.u32()
        if t == _BINARY_CLIPBOARD:
            for _ in range(r.u8()):
                r.skip(4)
                r.skip(r.u8())
                r.skip(r.u32())
            return r.pos, self._convert_clipboard(bytes(r.buf[:r.pos])), 0
        if t == 182:  # SubscribeUnixRelay
            r.skip(1)
            r.skip(r.u8())
            return r.pos, None, 0
        if t == 183:  # UnixRelay: name, then a length-prefixed payload
            r.skip(r.u8())
            payload = r.u32()
            return r.pos, None, payload
        if t == 248:  # ServerFence
            r.skip(7)
            r.skip(r.u8())
            return r.pos, None, 0
        raise _Desync(f"server message type {t}")

    def _after_rect(self) -> None:
        if self._rects_left is not None:
            self._rects_left -= 1
            if self._rects_left == 0:
                self._state = self._message
                return
        self._state = self._rect

    def _rect(self, r: _Reader) -> _Unit:
        # Read the whole unit before touching any state: a frame can end mid-header,
        # and the unit is then re-parsed from the start when more bytes arrive.
        x, y, w, h = r.u16(), r.u16(), r.u16(), r.u16()
        enc = r.s32()
        px = self._bpp
        if enc == -224:  # LastRect
            self._state = self._message
            return 12, None, 0
        hextile = None
        if enc == 0:  # Raw
            skip = w * h * px
        elif enc == 1:  # CopyRect
            skip = 4
        elif enc == 2:  # RRE: count, background, subrects
            skip = px + r.u32() * (px + 8)
        elif enc == 5:  # Hextile: tiles are parsed one by one
            skip = 0
            hextile = [x, y, w, h, 0, 0] if w and h else None
        elif enc == 7:
            skip = self._tight(r, w, h)
        elif enc == 16:  # ZRLE
            skip = r.u32()
        elif enc == -239:  # Cursor: pixels + bitmask
            skip = w * h * px + ((w + 7) // 8) * h
        elif enc == -240:  # XCursor
            skip = (6 + 2 * ((w + 7) // 8) * h) if w and h else 0
        elif enc in (-223, -258):  # DesktopSize, QEMUExtendedKeyEvent ack
            skip = 0
        elif enc == -308:  # ExtendedDesktopSize: count, 3 pad, 16 bytes per screen
            skip = 3 + r.u8() * 16
        else:
            raise _Desync(f"rect encoding {enc}")
        consumed = r.pos
        self._after_rect()
        if hextile:
            self._hextile = hextile
            self._state = self._hextile_tile
        return consumed, None, skip

    def _tight(self, r: _Reader, w: int, h: int) -> int:
        """Read a Tight rect's header; return the length of the data that follows it."""
        ctl = r.u8() >> 4
        if ctl == 0x08:  # fill
            return self._tpixel
        if 0x09 <= ctl <= 0x0D:  # JPEG, PNG, WebP, QOI, watermark: compact length + data
            return r.compact_len()
        if ctl > 0x0D:
            raise _Desync(f"tight subencoding {ctl:#x}")
        filter_id = r.u8() if ctl & 0x04 else 0
        if filter_id == 1:  # palette
            colours = r.u8() + 1
            r.skip(colours * self._tpixel)
            size = ((w + 7) // 8) * h if colours == 2 else w * h
        elif filter_id in (0, 2):  # copy, gradient
            size = w * h * self._tpixel
        else:
            raise _Desync(f"tight filter {filter_id}")
        # Below Tight's compression threshold the data is sent raw, without a length.
        return size if size < 12 else r.compact_len()

    def _hextile_tile(self, r: _Reader) -> _Unit:
        x, y, w, h, tx, ty = self._hextile  # type: ignore[misc]
        tw, th = min(16, w - tx), min(16, h - ty)
        px = self._bpp
        sub = r.u8()
        if sub & 1:  # Raw tile
            skip = tw * th * px
        else:
            if sub & 2:
                r.skip(px)  # background
            if sub & 4:
                r.skip(px)  # foreground
            n = r.u8() if sub & 8 else 0
            skip = n * (px + 2 if sub & 16 else 2)
        tx += 16
        if tx >= w:
            tx, ty = 0, ty + 16
        if ty >= h:
            self._hextile = None
            # the rect was already counted when its header was read
            self._state = self._message if self._rects_left == 0 else self._rect
        else:
            self._hextile = [x, y, w, h, tx, ty]
        return r.pos, None, skip
