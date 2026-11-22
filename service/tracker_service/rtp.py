"""RTP packets of a camera's video, put back together into whole pictures.

Each picture (access unit) is given with the RTP timestamp its packets carry,
untouched: that number is the camera's stream clock, the same clock Protect's
live reports name (clockStream), and it is what lets a box be tied to the
exact frame it was found in. Players and ffmpeg's RTSP reader rebase it, which
is why this is done here.

H.264 (RFC 6184) and H.265 (RFC 7798), the packetisations cameras use: whole
NAL units, aggregates, and fragments. Output is Annex B (start codes), which a
decoder takes as it is.

Plain Python, nothing else needed.
"""

from __future__ import annotations

from dataclasses import dataclass

START = b"\x00\x00\x00\x01"
RTP_WRAP = 1 << 32


@dataclass
class RtpPacket:
    payload_type: int
    marker: bool
    sequence: int
    timestamp: int
    payload: bytes


def parse_rtp(data: bytes) -> RtpPacket | None:
    """One RTP packet; None for anything that isn't one."""
    if len(data) < 12 or data[0] >> 6 != 2:
        return None
    padding = bool(data[0] & 0x20)
    extension = bool(data[0] & 0x10)
    start = 12 + 4 * (data[0] & 0x0F)
    if extension:
        if len(data) < start + 4:
            return None
        start += 4 + 4 * int.from_bytes(data[start + 2 : start + 4], "big")
    end = len(data)
    if padding and end > start:
        end -= data[-1]
    if end < start:
        return None
    return RtpPacket(
        payload_type=data[1] & 0x7F,
        marker=bool(data[1] & 0x80),
        sequence=int.from_bytes(data[2:4], "big"),
        timestamp=int.from_bytes(data[4:8], "big"),
        payload=bytes(data[start:end]),
    )


@dataclass
class AccessUnit:
    """One coded picture: its RTP timestamp (as sent, 0 .. 2^32-1) and bytes."""

    timestamp: int
    data: bytes
    key: bool
    whole: bool = True  # False when a packet of it was lost on the way


class Depacketizer:
    """Feed it the video packets in order; it gives back each picture when
    its last packet has come (the marker bit, or the next picture starting)."""

    def __init__(self, codec: str) -> None:
        codec = codec.lower()
        if codec not in ("h264", "h265"):
            raise ValueError(f"unsupported codec {codec}")
        self.h265 = codec == "h265"
        self._nals: list[bytes] = []
        self._fragment: bytearray | None = None
        self._timestamp: int | None = None
        self._sequence: int | None = None
        self._key = False
        self._whole = True

    def push(self, packet: RtpPacket) -> list[AccessUnit]:
        out: list[AccessUnit] = []
        if self._timestamp is not None and packet.timestamp != self._timestamp:
            unit = self._finish()  # its marker never came
            if unit:
                out.append(unit)
        if self._sequence is not None and packet.sequence != (self._sequence + 1) & 0xFFFF:
            self._whole = False
            self._fragment = None
        self._sequence = packet.sequence
        self._timestamp = packet.timestamp
        if packet.payload:
            (self._h265 if self.h265 else self._h264)(packet.payload)
        if packet.marker:
            unit = self._finish()
            if unit:
                out.append(unit)
        return out

    def _finish(self) -> AccessUnit | None:
        nals, self._nals = self._nals, []
        unit = None
        if nals and self._timestamp is not None:
            unit = AccessUnit(self._timestamp, b"".join(START + n for n in nals), self._key, self._whole and self._fragment is None)
        self._fragment = None
        self._timestamp = None
        self._key = False
        self._whole = True
        return unit

    def _add(self, nal: bytes) -> None:
        if not nal:
            return
        kind = (nal[0] >> 1) & 0x3F if self.h265 else nal[0] & 0x1F
        if (16 <= kind <= 21) if self.h265 else kind == 5:
            self._key = True
        self._nals.append(bytes(nal))

    def _h264(self, p: bytes) -> None:
        kind = p[0] & 0x1F
        if 1 <= kind <= 23:
            self._add(p)
        elif kind == 24:  # STAP-A: several small units in one packet
            at = 1
            while at + 2 <= len(p):
                size = int.from_bytes(p[at : at + 2], "big")
                self._add(p[at + 2 : at + 2 + size])
                at += 2 + size
        elif kind == 28 and len(p) >= 2:  # FU-A: a piece of a big unit
            first, last = bool(p[1] & 0x80), bool(p[1] & 0x40)
            if first:
                self._fragment = bytearray([(p[0] & 0xE0) | (p[1] & 0x1F)])
            if self._fragment is None:
                self._whole = False  # its beginning was lost
                return
            self._fragment += p[2:]
            if last:
                self._add(bytes(self._fragment))
                self._fragment = None

    def _h265(self, p: bytes) -> None:
        if len(p) < 2:
            return
        kind = (p[0] >> 1) & 0x3F
        if kind < 48:
            self._add(p)
        elif kind == 48:  # aggregation packet
            at = 2
            while at + 2 <= len(p):
                size = int.from_bytes(p[at : at + 2], "big")
                self._add(p[at + 2 : at + 2 + size])
                at += 2 + size
        elif kind == 49 and len(p) >= 3:  # fragmentation unit
            first, last = bool(p[2] & 0x80), bool(p[2] & 0x40)
            if first:
                self._fragment = bytearray([(p[0] & 0x81) | ((p[2] & 0x3F) << 1), p[1]])
            if self._fragment is None:
                self._whole = False
                return
            self._fragment += p[3:]
            if last:
                self._add(bytes(self._fragment))
                self._fragment = None


class Unwrap:
    """A 32-bit RTP timestamp as a number that keeps counting (it starts over
    every 13 hours at 90 kHz)."""

    def __init__(self) -> None:
        self._last: int | None = None

    def __call__(self, timestamp: int) -> int:
        if self._last is None:
            self._last = timestamp
            return timestamp
        step = (timestamp - self._last) % RTP_WRAP
        if step >= RTP_WRAP // 2:
            step -= RTP_WRAP  # a little back (pictures out of order), not almost a whole round forward
        self._last += step
        return self._last
