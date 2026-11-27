"""A made-up camera for the tests: a few frames of a square moving across a
dark picture, encoded, cut into RTP packets, and served over RTSP."""

from __future__ import annotations

import asyncio
from fractions import Fraction
import re

import numpy as np

W, H, SIDE = 320, 240, 40


def picture(i: int) -> np.ndarray:
    """Frame i: a white square whose left edge is at 10 * i."""
    img = np.zeros((H, W, 3), np.uint8)
    img[100:140, 10 * i : 10 * i + SIDE] = 255
    return img


def square_at(img: np.ndarray) -> int:
    """Which frame a decoded picture is (from where its square is)."""
    cols = np.where(img[120, :, 1] > 128)[0]
    return round(cols[0] / 10) if len(cols) else -1


def encode(codec: str, count: int) -> list[bytes]:
    """`count` frames, each as Annex B bytes."""
    import av

    ctx = av.CodecContext.create("libx265" if codec == "h265" else "libx264", "w")
    ctx.width, ctx.height, ctx.pix_fmt = W, H, "yuv420p"
    ctx.time_base = Fraction(1, 30)
    ctx.options = (
        {"preset": "ultrafast", "tune": "zerolatency", "x265-params": "bframes=0:log-level=none:repeat-headers=1:keyint=10"}
        if codec == "h265"
        else {"preset": "ultrafast", "tune": "zerolatency", "g": "10"}
    )
    out = []
    for i in range(count):
        frame = av.VideoFrame.from_ndarray(picture(i), format="bgr24").reformat(format="yuv420p")
        frame.pts = i
        out += [bytes(p) for p in ctx.encode(frame)]
    out += [bytes(p) for p in ctx.encode(None)]
    return out


def nals(annexb: bytes) -> list[bytes]:
    return [n for n in re.split(b"\x00\x00\x00\x01|\x00\x00\x01", annexb) if n]


def rtp(payload: bytes, seq: int, ts: int, marker: bool, pt: int = 96) -> bytes:
    return bytes([0x80, pt | (0x80 if marker else 0)]) + (seq & 0xFFFF).to_bytes(2, "big") + (ts & 0xFFFFFFFF).to_bytes(4, "big") + b"\x00\x00\x00\x01" + payload


def packetize(codec: str, frames: list[bytes], first_ts: int, step: int = 3000, mtu: int = 200, aggregate: bool = False) -> list[bytes]:
    """RTP packets for the frames: small units whole (or two to a packet with
    `aggregate`), big ones in fragments."""
    out, seq = [], 100
    for i, frame in enumerate(frames):
        ts = first_ts + i * step
        payloads: list[bytes] = []
        units = nals(frame)
        small = [u for u in units if len(u) <= mtu]
        if aggregate and len(small) >= 2:
            a, b = small[0], small[1]
            head = bytes([48 << 1, 1]) if codec == "h265" else bytes([24])
            payloads.append(head + len(a).to_bytes(2, "big") + a + len(b).to_bytes(2, "big") + b)
            units = [u for u in units if u is not a and u is not b]
        for unit in units:
            if len(unit) <= mtu:
                payloads.append(unit)
                continue
            if codec == "h265":
                kind, body = (unit[0] >> 1) & 0x3F, unit[2:]
                head = lambda s, e: bytes([(unit[0] & 0x81) | (49 << 1), unit[1], (0x80 if s else 0) | (0x40 if e else 0) | kind])  # noqa: E731
            else:
                kind, body = unit[0] & 0x1F, unit[1:]
                head = lambda s, e: bytes([(unit[0] & 0xE0) | 28, (0x80 if s else 0) | (0x40 if e else 0) | kind])  # noqa: E731
            parts = [body[k : k + mtu] for k in range(0, len(body), mtu)]
            for k, part in enumerate(parts):
                payloads.append(head(k == 0, k == len(parts) - 1) + part)
        for k, payload in enumerate(payloads):
            out.append(rtp(payload, seq, ts, k == len(payloads) - 1))
            seq += 1
    return out


class FakeCamera:
    """An RTSP server that plays the given RTP packets once, then hangs up."""

    def __init__(self, codec: str, packets: list[bytes], path: str = "/stream", pace: float = 0.0, port: int = 0) -> None:
        self.codec, self.packets, self.path, self.port = codec, packets, path, port
        self.pace = pace  # seconds between frames (0: everything at once)
        self.stay = False
        self.asked: list[str] = []
        self.gone = 0  # clients that have hung up, or been hung up on
        self.server: asyncio.AbstractServer | None = None

    async def __aenter__(self) -> FakeCamera:
        self.server = await asyncio.start_server(self._client, "127.0.0.1", self.port)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc) -> None:
        self.server.close()

    @property
    def url(self) -> str:
        return f"rtsp://127.0.0.1:{self.port}{self.path}"

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                raw = (await reader.readuntil(b"\r\n\r\n")).decode()
                method, target = raw.split()[0], raw.split()[1]
                cseq = re.search(r"CSeq: (\d+)", raw).group(1)
                self.asked.append(method)
                body = ""
                extra = ""
                if method == "DESCRIBE":
                    if not target.endswith(self.path):
                        writer.write(f"RTSP/1.0 404 Not Found\r\nCSeq: {cseq}\r\n\r\n".encode())
                        await writer.drain()
                        continue
                    name = "H265" if self.codec == "h265" else "H264"
                    body = f"v=0\r\ns=cam\r\nm=audio 0 RTP/AVP 97\r\na=rtpmap:97 MPEG4-GENERIC/16000\r\na=control:trackID=1\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 {name}/90000\r\na=control:trackID=0\r\n"
                    extra = f"Content-Base: {target}/\r\nContent-Type: application/sdp\r\n"
                elif method == "SETUP":
                    extra = "Session: 1234;timeout=60\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n"
                writer.write(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\n{extra}Content-Length: {len(body)}\r\n\r\n{body}".encode())
                await writer.drain()
                if method == "PLAY":
                    stamp = None
                    for i, packet in enumerate(self.packets):
                        if self.pace and stamp is not None and packet[4:8] != stamp:
                            await writer.drain()
                            await asyncio.sleep(self.pace)
                        stamp = packet[4:8]
                        writer.write(b"$\x00" + len(packet).to_bytes(2, "big") + packet)
                        if i % 7 == 0:  # something on the other channel now and then, to be skipped
                            writer.write(b"$\x01\x00\x04abcd")
                    await writer.drain()
                    if self.stay:  # the stream stays open until the client hangs up
                        await reader.read()
                    break
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.gone += 1
            writer.close()
