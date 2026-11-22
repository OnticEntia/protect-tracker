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

from .detect import Detector, FaceDetector, find_faces
from .follow import CameraFollow
from .rtp import Depacketizer
from .rtsp import RtspError, RtspReader, VideoTrack
from .video import Decoder, picture

log = logging.getLogger(__name__)

BEHIND = 4  # pictures waiting beyond which a frame is decoded but not looked at (to catch up)
WAITING = 600  # pictures and reports that may wait at most


class Engine:
    """The models, shared by the cameras (one look at a time)."""

    def __init__(self, models: Path, model: str = "m", faces: bool = True, min_conf: float = 0.3, face_conf: float = 0.6) -> None:
        self.detector = Detector(Path(models) / f"yolox_{model}.onnx")
        face_model = Path(models) / "face_yunet.onnx"
        self.face = FaceDetector(face_model) if faces and face_model.exists() else None
        self.model = model
        self.min_conf, self.face_conf = min_conf, face_conf
        self.size = (self.detector.w, self.detector.h)
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self.detector.device

    def look(self, frame) -> dict[str, list]:
        """What is in a decoded frame: {kind: [(x1, y1, x2, y2, score)]}."""
        small = picture(frame, *self.size)
        with self._lock:
            found = self.detector.detect(small, self.min_conf, ("person", "vehicle", "animal"))
            if self.face is not None and found["person"]:
                found["face"] = find_faces(self.face, picture(frame), found["person"], self.face_conf)
        return found


class CameraWorker(threading.Thread):
    def __init__(self, engine, stamp_ms: float, emit: Callable[[dict[str, Any]], None]) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        self.follow = CameraFollow(stamp_ms)
        self.emit = emit
        self.items: queue.Queue = queue.Queue(maxsize=WAITING)
        self.decoder: Decoder | None = None
        self.codec: str | None = None
        self.skipped = 0
        self.lost = 0
        self.look_ms: deque = deque(maxlen=300)
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
                self.emit(packet)
            return
        unit, at = item
        if self.decoder is None:
            return
        for rtp, frame in self.decoder.decode(unit):
            if self.items.qsize() > BEHIND:
                self.skipped += 1
                continue
            started = time.perf_counter()
            found = self.engine.look(frame)
            self.look_ms.append((time.perf_counter() - started) * 1000)
            self.frame_times.append(time.monotonic())
            for packet in self.follow.frame(rtp, found, at):
                self.emit(packet)

    def state(self) -> dict[str, Any]:
        times = list(self.frame_times)
        recent = [t for t in times if times and t > times[-1] - 5]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
        looks = sorted(self.look_ms)
        return {
            "codec": self.codec, "fps": round(fps, 1), "look_ms": round(looks[len(looks) // 2], 1) if looks else None,
            "skipped": self.skipped, "lost": self.lost, **self.follow.stats,
        }


class Camera:
    """A camera while it is watched."""

    def __init__(self, engine, source: str, stamp_ms: float, emit: Callable[[dict[str, Any]], None]) -> None:
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
