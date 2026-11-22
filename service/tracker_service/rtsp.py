"""A small RTSP reader: one camera's video, as RTP packets, over one TCP
connection (RTP interleaved in it, optionally inside TLS).

Small on purpose. It exists because the raw RTP timestamps are needed (see
rtp.py) and the usual readers do not give them. It does what a camera's or a
relay's RTSP server needs and no more: DESCRIBE, SETUP of the video track,
PLAY, and a keep-alive now and then.

The address may hold a key: it is never logged, and errors name the host only.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import ssl
from urllib.parse import urlsplit, urlunsplit

from .rtp import RtpPacket, parse_rtp

KEEPALIVE = 20.0  # seconds between keep-alives
TIMEOUT = 10.0  # seconds without anything before the connection counts as dead


class RtspError(Exception):
    """Couldn't get the video. Says what, never the address."""


@dataclass
class VideoTrack:
    codec: str  # "h264" | "h265"
    payload_type: int
    control: str
    clock_rate: int = 90_000
    parameter_sets: list[bytes] = field(default_factory=list)  # SPS/PPS (and VPS), as sent in the description


def parse_sdp(sdp: str) -> VideoTrack | None:
    """The first H.264/H.265 video track of a session description."""
    track: VideoTrack | None = None
    in_video = False
    payload = None
    for line in sdp.replace("\r", "").split("\n"):
        if line.startswith("m="):
            if track is not None:
                break
            parts = line[2:].split()
            in_video = parts[0] == "video"
            payload = int(parts[3]) if in_video and len(parts) > 3 and parts[3].isdigit() else None
            continue
        if not in_video or not line.startswith("a="):
            continue
        key, _, value = line[2:].partition(":")
        if key == "rtpmap":
            pt, _, what = value.partition(" ")
            name, _, rate = what.partition("/")
            name = name.upper()
            if pt.isdigit() and int(pt) == payload and name in ("H264", "H265", "HEVC"):
                track = track or VideoTrack("h264", int(pt), "")
                track.codec = "h264" if name == "H264" else "h265"
                track.payload_type = int(pt)
                track.clock_rate = int(rate.split("/")[0]) if rate.split("/")[0].isdigit() else 90_000
        elif key == "control":
            track = track or VideoTrack("h264", payload or 96, "")
            track.control = value.strip()
        elif key == "fmtp":
            track = track or VideoTrack("h264", payload or 96, "")
            for item in value.partition(" ")[2].split(";"):
                name, _, val = item.strip().partition("=")
                if name in ("sprop-parameter-sets", "sprop-vps", "sprop-sps", "sprop-pps"):
                    for one in val.split(","):
                        try:
                            track.parameter_sets.append(base64.b64decode(one + "=" * (-len(one) % 4)))
                        except ValueError:
                            pass
    return track if track is not None and track.payload_type is not None else None


def _without_userinfo(url: str) -> tuple[str, str | None]:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = host + (f":{parts.port}" if parts.port else "")
    auth = None
    if parts.username:
        auth = base64.b64encode(f"{parts.username}:{parts.password or ''}".encode()).decode()
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, "")), auth


class RtspReader:
    """`async with RtspReader(url) as r:` then `async for packet in r.packets()`."""

    def __init__(self, url: str, verify_tls: bool = False) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("rtsp", "rtsps") or not parts.hostname:
            raise RtspError("not an RTSP address")
        self._url, self._auth = _without_userinfo(url)
        self._host = parts.hostname
        self._tls = parts.scheme == "rtsps"
        self._port = parts.port or (322 if self._tls else 554)
        self._verify = verify_tls
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._cseq = 0
        self._session: str | None = None
        self.track: VideoTrack | None = None

    async def __aenter__(self) -> RtspReader:
        try:
            await self._open()
        except RtspError:
            await self.close()
            raise
        except (OSError, asyncio.TimeoutError, ssl.SSLError) as err:
            await self.close()
            raise RtspError(f"couldn't connect ({type(err).__name__})") from None
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        writer, self._writer = self._writer, None
        if writer is None:
            return
        try:
            if self._session:
                writer.write(self._request_bytes("TEARDOWN", self._url))
                await asyncio.wait_for(writer.drain(), 1)
        except Exception:  # noqa: BLE001 - it is going away anyway
            pass
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), 1)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ talking

    def _request_bytes(self, method: str, url: str, headers: dict[str, str] | None = None) -> bytes:
        self._cseq += 1
        lines = [f"{method} {url} RTSP/1.0", f"CSeq: {self._cseq}", "User-Agent: tracker"]
        if self._auth:
            lines.append(f"Authorization: Basic {self._auth}")
        if self._session:
            lines.append(f"Session: {self._session}")
        for key, value in (headers or {}).items():
            lines.append(f"{key}: {value}")
        return ("\r\n".join(lines) + "\r\n\r\n").encode()

    async def _response(self) -> tuple[int, dict[str, str], bytes]:
        """The next RTSP reply (interleaved packets before it are dropped)."""
        assert self._reader is not None
        while True:
            first = await asyncio.wait_for(self._reader.readexactly(1), TIMEOUT)
            if first == b"$":
                head = await asyncio.wait_for(self._reader.readexactly(3), TIMEOUT)
                await asyncio.wait_for(self._reader.readexactly(int.from_bytes(head[1:], "big")), TIMEOUT)
                continue
            return await self._reply_after(first)

    async def _reply_after(self, first: bytes) -> tuple[int, dict[str, str], bytes]:
        assert self._reader is not None
        raw = first + await asyncio.wait_for(self._reader.readuntil(b"\r\n\r\n"), TIMEOUT)
        lines = raw.decode("latin-1").split("\r\n")
        try:
            status = int(lines[0].split()[1])
        except (IndexError, ValueError):
            raise RtspError("an answer that isn't RTSP") from None
        headers = {}
        for line in lines[1:]:
            key, sep, value = line.partition(":")
            if sep:
                headers[key.strip().lower()] = value.strip()
        size = int(headers.get("content-length") or 0)
        body = await asyncio.wait_for(self._reader.readexactly(size), TIMEOUT) if size else b""
        return status, headers, body

    async def _ask(self, method: str, url: str, headers: dict[str, str] | None = None) -> tuple[dict[str, str], bytes]:
        assert self._writer is not None
        self._writer.write(self._request_bytes(method, url, headers))
        await self._writer.drain()
        status, reply, body = await self._response()
        if status != 200:
            raise RtspError(f"{method} was refused ({status})")
        return reply, body

    async def _open(self) -> None:
        context = None
        if self._tls:
            context = ssl.create_default_context()
            if not self._verify:  # a console's own certificate is self-signed
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port, ssl=context), TIMEOUT
        )
        reply, body = await self._ask("DESCRIBE", self._url, {"Accept": "application/sdp"})
        self.track = parse_sdp(body.decode("utf-8", "replace"))
        if self.track is None:
            raise RtspError("no H.264 or H.265 video in the stream")
        base = (reply.get("content-base") or reply.get("content-location") or self._url).rstrip("/")
        control = self.track.control
        if control.startswith(("rtsp://", "rtsps://")):
            target = control
        elif control in ("", "*"):
            target = base
        else:
            target = f"{base}/{control}"
        reply, _ = await self._ask("SETUP", target, {"Transport": "RTP/AVP/TCP;unicast;interleaved=0-1"})
        self._session = (reply.get("session") or "").split(";")[0] or None
        await self._ask("PLAY", base, {"Range": "npt=0.000-"})

    # ------------------------------------------------------------ reading

    async def packets(self) -> AsyncIterator[RtpPacket]:
        """The video's RTP packets, as they come, until the stream ends."""
        assert self._reader is not None and self._writer is not None and self.track is not None
        loop = asyncio.get_running_loop()
        asked = loop.time()
        while True:
            try:
                first = await asyncio.wait_for(self._reader.readexactly(1), TIMEOUT)
                if first != b"$":
                    await self._reply_after(first)  # the answer to a keep-alive
                    continue
                head = await asyncio.wait_for(self._reader.readexactly(3), TIMEOUT)
                data = await asyncio.wait_for(self._reader.readexactly(int.from_bytes(head[1:], "big")), TIMEOUT)
            except asyncio.IncompleteReadError:
                return
            except asyncio.TimeoutError:
                raise RtspError("the stream went quiet") from None
            if head[0] == 0:
                packet = parse_rtp(data)
                if packet is not None and packet.payload_type == self.track.payload_type:
                    yield packet
            if loop.time() - asked > KEEPALIVE:
                asked = loop.time()
                self._writer.write(self._request_bytes("OPTIONS", self._url))
                await self._writer.drain()
