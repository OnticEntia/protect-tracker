"""Coded pictures to frames, each still carrying its RTP timestamp.

Frames are held in memory only for as long as it takes to look at them;
nothing is ever written to disk.
"""

from __future__ import annotations

import threading

import numpy as np

from .rtp import START, AccessUnit, Unwrap


MOST = (1280, 720)  # a picture larger than this is brought down to it as it is decoded: nothing looks closer


def fits(width: int, height: int, most: tuple[int, int] = MOST) -> tuple[int, int] | None:
    """The size a picture is brought down to (its shape kept, even numbers),
    or None if it is small enough as it is."""
    if width <= most[0] and height <= most[1]:
        return None
    scale = min(most[0] / width, most[1] / height)
    return max(2, int(round(width * scale / 2)) * 2), max(2, int(round(height * scale / 2)) * 2)


class Decoder:
    """Decodes access units (rtp.py) with FFmpeg's decoders through PyAV. A
    frame comes back as (RTP timestamp, frame); the timestamp is the one its
    packets carried. `picture` turns a frame into pixels at the size wanted.

    A stream no larger than MOST (the medium stream of a camera) is decoded
    in software, as it comes. A larger one (a camera's main stream: 3840 x
    2160) costs several times as much to decode and as much again to shrink
    for every look, more than one thread has to give at thirty frames a
    second; and nothing here looks closer than MOST. So it is brought down to
    MOST as it is decoded:

      on the graphics card where that can be had (its video decoder, which
      also does the shrinking: `how` is "gpu"). The frame of a packet then
      comes with the next packet, one frame later; `came` gives back the
      time its own packet arrived.
      else in software, shrunk once right after decoding ("software").

    Which it is to be is settled at the stream's first key picture, which is
    decoded in software to learn the size. `gpu=False` keeps to software."""

    def __init__(self, codec: str, parameter_sets: list[bytes] | None = None, most: tuple[int, int] | None = MOST, gpu: bool = True) -> None:
        import av  # noqa: PLC0415

        self._av = av
        self._name = "hevc" if codec == "h265" else "h264"
        self._ctx = self._software()
        self._head = b"".join(START + p for p in parameter_sets or [])
        self._unwrap = Unwrap()
        self._started = False
        self._most, self._gpu = most, gpu
        self._down: tuple[int, int] | None = None  # the size frames are shrunk to after decoding (software)
        self._came: dict[int, float] = {}  # RTP timestamp -> when its packet arrived (kept while its frame is awaited)
        self.how = "software"
        self.size: tuple[int, int] | None = None  # the stream's own size
        self.gpu_error: str | None = None  # why the card's decoder was not used, if it was tried

    def _software(self):
        ctx = self._av.CodecContext.create(self._name, "r")
        ctx.thread_type = "SLICE"  # frame threads would hold pictures back
        ctx.thread_count = 2
        return ctx

    def _on_card(self, size: tuple[int, int]):
        """The card's decoder, giving frames of `size`. Raises if there is none."""
        ctx = self._av.CodecContext.create(f"{self._name}_cuvid", "r")
        ctx.options = {"resize": f"{size[0]}x{size[1]}"}
        ctx.flags |= self._av.codec.context.Flags.low_delay  # (else it holds four frames back)
        return ctx

    def _packet(self, data: bytes, pts: int):
        packet = self._av.Packet(data)
        packet.pts = pts
        return packet

    def _settle(self, data: bytes, pts: int, frames: list, card: bool = True) -> list:
        """The first key picture has been decoded (in software): is the
        stream too large, and if so, how is it brought down?"""
        first = frames[0]
        self.size = (first.width, first.height)
        down = fits(first.width, first.height, self._most) if self._most else None
        if down is None:
            return frames
        if self._gpu and card:
            try:
                ctx = self._on_card(down)
                frames = ctx.decode(self._packet(data, pts))  # (the same key picture again; its frame comes with the next packet)
                self._ctx, self.how = ctx, "gpu"
                return frames
            except Exception as err:  # noqa: BLE001 - no such decoder, no card, no room on it: software does it
                self.gpu_error = type(err).__name__
        self._down = down
        return frames

    def came(self, rtp: int, default: float) -> float:
        """When the packet of the frame with this timestamp arrived."""
        return self._came.pop(rtp, default)

    def decode(self, unit: AccessUnit, at: float | None = None) -> list[tuple[int, object]]:
        if not self._started:
            if not unit.key:
                return []  # nothing can be made of a picture that leans on ones not seen
            data = self._head + unit.data
        else:
            data = unit.data
        pts = self._unwrap(unit.timestamp)
        try:
            frames = self._ctx.decode(self._packet(data, pts))
            self._started = True
            if frames and self.size is None:
                # (a first frame that did not come with the key picture's own packet: software has it from here)
                frames = self._settle(data, pts, frames, card=unit.key)
        except self._av.error.FFmpegError:
            self._started = False  # start again at the next key picture
            if self.how == "gpu":  # (in software from here on: a decoder that fails once is not tried again on this stream)
                self.how, self.gpu_error, self._ctx = "software", "failed while decoding", self._software()
                self._down = fits(*self.size, self._most) if self.size and self._most else None
            return []
        if at is not None and self.how == "gpu":
            self._came[pts % (1 << 32)] = at
            if len(self._came) > 64:  # (frames that never came)
                for key in list(self._came)[:-16]:
                    del self._came[key]
        out = []
        for frame in frames:
            if frame.pts is None:
                continue
            stamp = int(frame.pts) % (1 << 32)
            if self._down is not None:
                frame = scaled(frame, self._down[0], self._down[1], None, "AREA")
            out.append((stamp, frame))
        return out


_scalers = threading.local()


def scaled(frame, width: int | None = None, height: int | None = None, format: str | None = None, interpolation: str | None = None):
    """A frame at another size or in another form. The scaler is kept and
    used again for the next frame (one per thread and kind of result):
    setting one up costs many times what using it does, and a frame's own
    `reformat` and `to_ndarray` set up a new one for every frame."""
    from av.video.reformatter import VideoReformatter  # noqa: PLC0415

    kept = _scalers.__dict__.setdefault("kept", {})
    key = (width, height, format, interpolation)
    scaler = kept.get(key)
    if scaler is None:
        scaler = kept[key] = VideoReformatter()
    wanted = {name: value for name, value in (("width", width), ("height", height), ("format", format), ("interpolation", interpolation)) if value}
    return scaler.reformat(frame, **wanted)


def picture(frame, width: int | None = None, height: int | None = None) -> np.ndarray:
    """A decoded frame as h x w x 3 BGR pixels, scaled when a size is given."""
    return scaled(frame, width, height, "bgr24").to_ndarray()


def small_grey(frame, width: int, height: int) -> np.ndarray:
    """A decoded frame as a small grey picture (h x w), each point the mean of what it covers."""
    return scaled(frame, width, height, "gray", "AREA").to_ndarray()
