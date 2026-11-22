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
from .pace import MIN_FPS, Pace
from .rtp import Depacketizer
from .rtsp import RtspError, RtspReader, VideoTrack
from .video import Decoder, picture

log = logging.getLogger(__name__)

IDLE_EVERY = 1  # while nothing moves, one frame in this many is looked at. 1: every frame, always (the first
# sight of someone is then never later than a frame). More is only taken as far as the floor allows (pace.py)
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
                 idle_every: int = IDLE_EVERY, far_every: int = FAR_EVERY, min_fps: float = MIN_FPS) -> None:
        self.idle_every, self.far_every, self.min_fps = max(1, int(idle_every)), max(1, int(far_every)), float(min_fps)
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

    def look(self, frame, far: bool = False, faces: bool = True) -> dict[str, list]:
        """What is in a decoded frame: {kind: [(x1, y1, x2, y2, score)]}.
        `far`: also the sharper look for people far off (far_people);
        `faces`: also the faces of the people found."""
        small = picture(frame, *self.size)
        with self._lock:
            found = self.detector.detect(small, self.min_conf, ("person", "vehicle", "animal"))
            if far and self.far:
                found["person"] = merge_far(found["person"], self.far_people(frame))
            if faces and self.face is not None and found["person"]:
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
        self.pace = Pace(getattr(engine, "min_fps", MIN_FPS), getattr(engine, "idle_every", IDLE_EVERY), getattr(engine, "far_every", FAR_EVERY))
        self.lost = 0
        self.look_ms: deque = deque(maxlen=300)
        self.far_ms: deque = deque(maxlen=300)  # the looks that include the sharper one for people far off
        self._pace_state = self.pace.state(0.0)

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
                self.emit(self._rated(packet), "auto")
            if self.follow.last_report is not None:
                self.check.report(*self.follow.last_report)
            return
        unit, at = item
        if self.decoder is None:
            return
        for rtp, frame in self.decoder.decode(unit):
            now = time.monotonic()
            plan = self.pace.plan(max(0.0, time.time() - at / 1000.0), self.follow.busy(), now)
            self._pace_state = self.pace.state(now)
            if not plan["look"]:
                continue
            started = time.perf_counter()
            found = self.engine.look(frame, plan["far"], plan["faces"])
            (self.far_ms if plan["far"] else self.look_ms).append((time.perf_counter() - started) * 1000)
            self.follow.last_frame = None
            for packet in self.follow.frame(rtp, found, at):
                self.emit(self._rated(packet), "auto")  # in Protect's name: what Protect reported, refined
            for packet in self.follow.local_packets:
                self.emit(self._rated(packet), "local")  # on our own word
            if self.follow.last_frame is not None:  # the picture check: numbers only, the small picture is dropped again
                when, ours = self.follow.last_frame
                self.check.frame(when, picture(frame, CHECK_W, CHECK_H)[:, :, 1], ours)

    def _rated(self, packet: dict[str, Any]) -> dict[str, Any]:
        """A packet with how the camera is being kept up with: frames looked
        at a second, the floor, and whether that floor is being held."""
        pace = self._pace_state
        packet["fps"], packet["fpsFloor"], packet["fpsOk"] = pace["fps"], pace["floor"], not pace["below"]
        return packet

    def state(self) -> dict[str, Any]:
        looks, fars = sorted(self.look_ms), sorted(self.far_ms)
        pace = self.pace.state(time.monotonic())
        return {
            "codec": self.codec, **pace, "look_ms": round(looks[len(looks) // 2], 1) if looks else None,
            "far_look_ms": round(fars[len(fars) // 2], 1) if fars else None,
            "lost": self.lost, **self.follow.stats,
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
