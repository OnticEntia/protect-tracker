"""Coded pictures to frames, each still carrying its RTP timestamp."""

from __future__ import annotations

import numpy as np

from .rtp import START, AccessUnit, Unwrap


class Decoder:
    """Decodes access units (rtp.py) with FFmpeg's decoder through PyAV. A
    frame comes back as (RTP timestamp, h x w x 3 BGR picture); the timestamp
    is the one its packets carried."""

    def __init__(self, codec: str, parameter_sets: list[bytes] | None = None) -> None:
        import av  # noqa: PLC0415

        self._av = av
        self._ctx = av.CodecContext.create("hevc" if codec == "h265" else "h264", "r")
        self._ctx.thread_type = "SLICE"  # frame threads would hold pictures back
        self._ctx.thread_count = 2
        self._head = b"".join(START + p for p in parameter_sets or [])
        self._unwrap = Unwrap()
        self._started = False

    def decode(self, unit: AccessUnit, width: int | None = None, height: int | None = None) -> list[tuple[int, np.ndarray]]:
        if not self._started:
            if not unit.key:
                return []  # nothing can be made of a picture that leans on ones not seen
            self._started = True
            data = self._head + unit.data
        else:
            data = unit.data
        packet = self._av.Packet(data)
        packet.pts = self._unwrap(unit.timestamp)
        out = []
        try:
            frames = self._ctx.decode(packet)
        except self._av.error.FFmpegError:
            self._started = False  # start again at the next key picture
            return []
        for frame in frames:
            if frame.pts is None:
                continue
            kw = {"width": width, "height": height} if width and height else {}
            out.append((int(frame.pts) % (1 << 32), frame.to_ndarray(format="bgr24", **kw)))
        return out
