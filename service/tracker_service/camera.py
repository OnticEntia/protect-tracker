"""One camera being watched: its video read, looked at, and followed.

The video is read on the event loop (rtsp.py) and handed, with Protect's
reports for the camera, to a thread of its own that decodes, runs the
detectors and does the following (follow.py). What comes out is passed back
through `emit`. Everything is in memory; nothing is written anywhere.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np

from .check import H as CHECK_H, W as CHECK_W, PictureCheck
from .detect import Detector, FaceDetector, find_faces, iou
from .follow import CameraFollow
from .rtp import Depacketizer
from .rtsp import RtspError, RtspReader, VideoTrack
from .video import Decoder, picture

log = logging.getLogger(__name__)

BEHIND = 4  # pictures waiting beyond which a frame is decoded but not looked at (to catch up)
IDLE_EVERY = 1  # while nothing moves, one frame in this many is looked at. 1: every frame, always (the first
# sight of someone is then never later than a frame); 6 is five looks a second, for a machine with less to spare
FAR_EVERY = 6  # one frame in this many also gets the sharper look for people far off
FAR_SIZE = (1280, 720)  # the picture that look is made on, cut into four tiles of the detector's own size
FAR_CONF = 0.6  # what a sighting in a tile needs (a tile shows less of the scene to judge by)
WAITING = 600  # pictures and reports that may wait at most


def _base36(n: int) -> str:
    digits, out = "0123456789abcdefghijklmnopqrstuvwxyz", ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out or "0"


class Engine:
    """The models, shared by the cameras (one look at a time)."""

    def __init__(self, models: Path, model: str = "m", faces: bool = True, min_conf: float = 0.3, face_conf: float = 0.6, far: bool = True,
                 idle_every: int = IDLE_EVERY, far_every: int = FAR_EVERY) -> None:
        self.idle_every, self.far_every = max(1, int(idle_every)), max(1, int(far_every))
        self.detector = Detector(Path(models) / f"yolox_{model}.onnx")
        face_model = Path(models) / "face_yunet.onnx"
        self.face = FaceDetector(face_model) if faces and face_model.exists() else None
        self.model = model
        self.min_conf, self.face_conf = min_conf, face_conf
        self.size = (self.detector.w, self.detector.h)
        self.far = far and self.size == (FAR_SIZE[0] // 2, FAR_SIZE[1] // 2)  # the tiles must be the detector's size
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self.detector.device

    def look(self, frame, far: bool = False) -> dict[str, list]:
        """What is in a decoded frame: {kind: [(x1, y1, x2, y2, score)]}.
        `far`: also the sharper look for people far off (far_people)."""
        small = picture(frame, *self.size)
        with self._lock:
            found = self.detector.detect(small, self.min_conf, ("person", "vehicle", "animal"))
            if far and self.far:
                found["person"] = merge_far(found["person"], self.far_people(frame))
            if self.face is not None and found["person"]:
                found["face"] = find_faces(self.face, picture(frame), found["person"], self.face_conf)
        return found

    def far_people(self, frame) -> list:
        """People far off: the picture at twice the detector's size, cut into
        four tiles, each looked at on its own, so someone far away is four
        times the pixels they are in the whole-picture look."""
        big = picture(frame, *FAR_SIZE)
        w, h = self.size
        out = []
        for ty in (0, h):
            for tx in (0, w):
                tile = np.ascontiguousarray(big[ty : ty + h, tx : tx + w])
                for x1, y1, x2, y2, score in self.detector.detect(tile, FAR_CONF, ("person",))["person"]:
                    out.append(((tx + x1 * w) / FAR_SIZE[0], (ty + y1 * h) / FAR_SIZE[1], (tx + x2 * w) / FAR_SIZE[0], (ty + y2 * h) / FAR_SIZE[1], score))
        return out


def merge_far(people: list, far: list) -> list:
    """The whole-picture look's people, and those only the sharper look found."""
    out = list(people)
    for box in far:
        if all(iou(box, other) < 0.3 for other in out):
            out.append(box)
    return out


class CameraWorker(threading.Thread):
    def __init__(self, engine, stamp_ms: float, emit: Callable[[dict[str, Any], str], None]) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        # (the time it started is part of its own track ids, so they are not used again after a restart)
        self.follow = CameraFollow(stamp_ms, epoch=_base36(int(time.time())))
        self.check = PictureCheck()
        self.emit = emit
        self.items: queue.Queue = queue.Queue(maxsize=WAITING)
        self.decoder: Decoder | None = None
        self.codec: str | None = None
        self.skipped = 0
        self.idle_skipped = 0
        self.lost = 0
        self._count = 0
        self.look_ms: deque = deque(maxlen=300)
        self.far_ms: deque = deque(maxlen=300)  # the looks that include the sharper one for people far off
        self.frame_times: deque = deque(maxlen=300)

    def put(self, item) -> bool:
        try:
            self.items.put_nowait(item)
            return True
        except queue.Full:
            self.lost += 1
            return False

    def stop(self) -> None:
        while True:  # whatever waits is of no use any more
            try:
                self.items.get_nowait()
            except queue.Empty:
                break
        self.items.put(None)

    def run(self) -> None:
        while True:
            item = self.items.get()
            if item is None:
                return
            try:
                self._one(item)
            except Exception:  # noqa: BLE001 - one bad picture or report is no reason to stop watching
                log.debug("A picture or report couldn't be used", exc_info=True)

    def _one(self, item) -> None:
        if isinstance(item, VideoTrack):  # a stream has just been opened
            self.decoder, self.codec = Decoder(item.codec, item.parameter_sets), item.codec
            return
        if item[0] == "report":
            for packet in self.follow.report(item[1], item[2]):
                self.emit(packet, "auto")
            if self.follow.last_report is not None:
                self.check.report(*self.follow.last_report)
            return
        unit, at = item
        if self.decoder is None:
            return
        for rtp, frame in self.decoder.decode(unit):
            if self.items.qsize() > BEHIND:
                self.skipped += 1
                continue
            self._count += 1
            if self._count % getattr(self.engine, "idle_every", IDLE_EVERY) and not self.follow.busy():
                self.idle_skipped += 1  # nothing moves: a look now and then is enough to notice when something does
                continue
            far = self._count % getattr(self.engine, "far_every", FAR_EVERY) == 0
            started = time.perf_counter()
            found = self.engine.look(frame, far)
            (self.far_ms if far else self.look_ms).append((time.perf_counter() - started) * 1000)
            self.frame_times.append(time.monotonic())
            self.follow.last_frame = None
            for packet in self.follow.frame(rtp, found, at):
                self.emit(packet, "auto")  # in Protect's name: what Protect reported, refined
            for packet in self.follow.local_packets:
                self.emit(packet, "local")  # on our own word
            if self.follow.last_frame is not None:  # the picture check: numbers only, the small picture is dropped again
                when, ours = self.follow.last_frame
                self.check.frame(when, picture(frame, CHECK_W, CHECK_H)[:, :, 1], ours)

    def state(self) -> dict[str, Any]:
        times = list(self.frame_times)
        recent = [t for t in times if times and t > times[-1] - 5]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
        looks, fars = sorted(self.look_ms), sorted(self.far_ms)
        return {
            "codec": self.codec, "fps": round(fps, 1), "look_ms": round(looks[len(looks) // 2], 1) if looks else None,
            "far_look_ms": round(fars[len(fars) // 2], 1) if fars else None,
            "skipped": self.skipped, "idle_skipped": self.idle_skipped, "lost": self.lost, **self.follow.stats,
            "check": self.check.result(), "head_start": self.follow.head_start(), "local": dict(self.follow.local.stats),
        }


class Camera:
    """A camera while it is watched."""

    def __init__(self, engine, source: str, stamp_ms: float, emit: Callable[[dict[str, Any], str], None]) -> None:
        self.worker = CameraWorker(engine, stamp_ms, emit)
        self.worker.start()
        self.error: str | None = None
        self.opened = 0
        self._task = asyncio.create_task(self._read(source))

    def report(self, packet: dict[str, Any]) -> None:
        self.worker.put(("report", packet, time.time() * 1000))

    async def stop(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self.worker.stop()

    async def _read(self, source: str) -> None:
        """Keep reading the video; start over after a drop."""
        backoff = 1.0
        while True:
            try:
                async with RtspReader(source) as reader:
                    self.opened += 1
                    self.error = None
                    backoff = 1.0
                    self.worker.put(reader.track)
                    depack = Depacketizer(reader.track.codec)
                    async for packet in reader.packets():
                        for unit in depack.push(packet):
                            self.worker.put((unit, time.time() * 1000))
                self.error = "the stream ended"
            except RtspError as err:
                self.error = str(err)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    def state(self) -> dict[str, Any]:
        return {"error": self.error, "opened": self.opened, **self.worker.state()}
