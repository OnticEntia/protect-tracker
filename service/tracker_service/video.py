"""Coded pictures to frames, each still carrying its RTP timestamp.

Frames are held in memory only for as long as it takes to look at them;
nothing is ever written to disk.
"""

from __future__ import annotations

import numpy as np

from .rtp import START, AccessUnit, Unwrap


class Decoder:
    """Decodes access units (rtp.py) with FFmpeg's decoder through PyAV. A
    frame comes back as (RTP timestamp, frame); the timestamp is the one its
    packets carried. `picture` turns a frame into pixels at the size wanted."""

    def __init__(self, codec: str, parameter_sets: list[bytes] | None = None) -> None:
        import av  # noqa: PLC0415

        self._av = av
        self._ctx = av.CodecContext.create("hevc" if codec == "h265" else "h264", "r")
        self._ctx.thread_type = "SLICE"  # frame threads would hold pictures back
        self._ctx.thread_count = 2
        self._head = b"".join(START + p for p in parameter_sets or [])
        self._unwrap = Unwrap()
        self._started = False

    def decode(self, unit: AccessUnit) -> list[tuple[int, object]]:
        if not self._started:
            if not unit.key:
                return []  # nothing can be made of a picture that leans on ones not seen
            self._started = True
            data = self._head + unit.data
        else:
            data = unit.data
        packet = self._av.Packet(data)
        packet.pts = self._unwrap(unit.timestamp)
        try:
            frames = self._ctx.decode(packet)
        except self._av.error.FFmpegError:
            self._started = False  # start again at the next key picture
            return []
        return [(int(frame.pts) % (1 << 32), frame) for frame in frames if frame.pts is not None]


def picture(frame, width: int | None = None, height: int | None = None) -> np.ndarray:
    """A decoded frame as h x w x 3 BGR pixels, scaled when a size is given."""
    if width and height:
        return frame.to_ndarray(format="bgr24", width=width, height=height)
    return frame.to_ndarray(format="bgr24")
