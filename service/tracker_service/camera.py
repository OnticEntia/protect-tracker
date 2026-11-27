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

from .bench import Bench
from .check import H as CHECK_H, W as CHECK_W, PictureCheck
from .detect import Detector, FaceDetector, find_faces
from .follow import CameraFollow
from .pace import MIN_FPS, Pace
from .rtp import Depacketizer
from .rtsp import RtspError, RtspReader, VideoTrack
from .video import Decoder, picture, small_grey
from .view import EVERY_MS as VIEW_EVERY_MS, H as VIEW_H, MOVING, W as VIEW_W, ViewMotion

log = logging.getLogger(__name__)

IDLE_EVERY = 1  # while nothing moves, one frame in this many is looked at. 1: every frame, always (the first
# sight of someone is then never later than a frame). More is only taken as far as the floor allows (pace.py)
FAR_EVERY = 1  # one frame in this many gets the sharp look (the picture at twice the size) in place of the ordinary one
FACES_AT_ONCE = 3  # the faces of this many people (the biggest in the picture) are looked for in a frame
WIDE = "wide1280"  # the folder (in the models folder) of the detector's copy for that: a picture 1280 across
SAID_AN_HOUR = 20  # moves of the view said in the log in an hour, at most
BELOW_SAID = 10.0  # under the frame-rate floor for this long (seconds): said in the log
WAITING = 600  # pictures and reports that may wait at most
RETRY_FIRST, RETRY_MOST = 1.0, 60.0  # a stream that cannot be read is tried again after this long, then twice as long each time, at most this
DOWN_AFTER = 10.0  # no video for this long: the stream counts as down (a drop that mends itself in a moment is not worth a word)
RAN = 10.0  # a stream that ran this long was a good one: after it the waits start short again


def _median(values) -> float | None:
    values = sorted(values)
    return round(values[len(values) // 2], 2) if values else None


def _base36(n: int) -> str:
    digits, out = "0123456789abcdefghijklmnopqrstuvwxyz", ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out or "0"


class Engine:
    """The models, shared by the cameras (one look at a time)."""

    def __init__(self, models: Path, model: str = "m", faces: bool = True, min_conf: float = 0.3, face_conf: float = 0.6, far: bool = True,
                 idle_every: int = IDLE_EVERY, far_every: int = FAR_EVERY, min_fps: float = MIN_FPS, gpu_decode: bool = True) -> None:
        self.gpu_decode = gpu_decode  # a stream larger than is looked at is decoded on the card where that can be had (video.py)
        self.idle_every, self.far_every, self.min_fps = max(1, int(idle_every)), max(1, int(far_every)), float(min_fps)
        self.detector = Detector(Path(models) / f"yolox_{model}.onnx")
        face_model = Path(models) / "face_yunet.onnx"
        self.face = FaceDetector(face_model) if faces and face_model.exists() else None
        self.model = model
        self.min_conf, self.face_conf = min_conf, face_conf
        self.size = (self.detector.w, self.detector.h)
        # the sharp look: the same model made for a picture twice as wide (tools/make_fast_models.py --width 1280),
        # where that file is there. Someone far off is four times the pixels in it; so is everything else.
        wide = Path(models) / WIDE / f"yolox_{model}.onnx"
        self.wide = Detector(wide) if far and wide.exists() else None
        self.far = self.wide is not None
        self.bench = Bench(Path(models))
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self.detector.device

    def look(self, frame, far: bool = False, faces: bool = True) -> dict[str, list]:
        """What is in a decoded frame: {kind: [(x1, y1, x2, y2, score)]}.
        `far`: the sharp look, at twice the size, in place of the ordinary one;
        `faces`: also the faces of the people found."""
        detector = self.wide if far and self.wide is not None else self.detector
        scaled = picture(frame, detector.w, detector.h)
        with self._lock:
            found = detector.detect(scaled, self.min_conf, ("person", "vehicle", "animal"))
            if faces and self.face is not None and found["person"]:
                found["face"] = find_faces(self.face, picture(frame), found["person"], self.face_conf, most=FACES_AT_ONCE)
        return found


class ViewWatch(threading.Thread):
    """Is the camera's whole view moving (view.py), asked on a thread of its
    own. The camera's thread only makes the small picture (a twentieth of a
    millisecond) and leaves it here; the comparing, which is many small steps
    that each may have to wait their turn beside the other threads, then
    holds up no frame. The answer (`state`) is a few milliseconds behind the
    picture it is of, which is nothing to a check made ten times a second.
    Only the newest picture left is looked at: one not yet looked at when the
    next comes is dropped.

    Each time the view starts moving and each time it has come to rest is a
    line in the log (with how long it moved), so that what the camera did can
    be read there afterwards. A camera that moves all day (one on patrol)
    would fill the log: past SAID_AN_HOUR moves in an hour that is said once
    and the lines stop, until an hour with fewer."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.motion = ViewMotion()
        self._moving_since: float | None = None  # the time of the picture in which it began to move
        self._said: deque = deque()  # when (our clock) the last moves were said
        self._quiet = False  # moving too often to say each time
        self.ms: deque = deque(maxlen=300)  # how long a comparison took, waiting included
        self.skipped = 0
        self._newest: tuple[float, Any] | None = None
        self._offered: float | None = None
        self._wake = threading.Event()
        self._stopped = False

    @property
    def state(self) -> str:
        return self.motion.state

    def due(self, when: float) -> bool:
        return self._offered is None or when - self._offered >= VIEW_EVERY_MS or when < self._offered

    def offer(self, when: float, grey) -> None:
        if self._newest is not None:
            self.skipped += 1
        self._offered, self._newest = when, (when, grey)
        self._wake.set()

    def afresh(self) -> None:
        """The camera's clock started over: nothing held means anything."""
        self.motion = ViewMotion()
        self._moving_since = None

    def check(self, when: float, grey) -> str:
        """Put one small picture to the question, and say in the log when the answer changes."""
        was = self.motion.state
        state = self.motion.frame(when, grey)
        if state == MOVING and was != MOVING:
            self._moving_since = when
            now = time.monotonic()
            while self._said and now - self._said[0] > 3600:
                self._said.popleft()
            self._said.append(now)
            if len(self._said) <= SAID_AN_HOUR:
                self._quiet = False
                log.info("A camera's view is moving")
            elif not self._quiet:
                self._quiet = True
                log.info("A camera's view moves often (more than %d times this hour): not said each time until it does so less; "
                         "the count is in the status", SAID_AN_HOUR)
        elif was == MOVING and state != MOVING:
            if not self._quiet:
                moved = (when - self._moving_since) / 1000.0 if self._moving_since is not None and when >= self._moving_since else 0.0
                log.info("A camera's view is steady again (it moved for %.1f s)", moved)
            self._moving_since = None
        return state

    def stop(self) -> None:
        self._stopped = True
        self._wake.set()

    def run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stopped:
                return
            newest, self._newest = self._newest, None
            if newest is None:
                continue
            started = time.perf_counter()
            try:
                self.check(*newest)
            except Exception:  # noqa: BLE001 - one bad picture is no reason to stop asking
                log.debug("A view check failed", exc_info=True)
            self.ms.append((time.perf_counter() - started) * 1000)


class CameraWorker(threading.Thread):
    def __init__(self, engine, stamp_ms: float, emit: Callable[[dict[str, Any], str], None]) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        # (the time it started is part of its own track ids, so they are not used again after a restart)
        self.stamp_ms = stamp_ms
        self.follow = CameraFollow(stamp_ms, epoch=_base36(int(time.time())))
        self._reopened = False
        self.check = PictureCheck()
        self.view = ViewWatch()  # is the whole view moving (a camera that turns or zooms): asked beside this thread
        self.view_ms: deque = deque(maxlen=300)  # what asking costs this thread, each time (making the small picture)
        self.emit = emit
        self.items: queue.Queue = queue.Queue(maxsize=WAITING)
        self.decoder: Decoder | None = None
        self.codec: str | None = None
        self.pace = Pace(getattr(engine, "min_fps", MIN_FPS), getattr(engine, "idle_every", IDLE_EVERY), getattr(engine, "far_every", FAR_EVERY))
        self.lost = 0
        self.restarts = 0  # times the following started afresh because the camera's clock did
        self.look_ms: deque = deque(maxlen=300)
        self._below_since: float | None = None  # since when the floor has not been held
        self._below_said = False
        self.decode_ms: deque = deque(maxlen=300)  # what decoding a frame takes (and bringing a large one down)
        self.far_ms: deque = deque(maxlen=300)  # the looks that include the sharper one for people far off
        self._pace_state = self.pace.state(0.0)

    def start(self) -> None:
        self.view.start()
        super().start()

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
                self.view.stop()
                return
            try:
                self._one(item)
            except Exception:  # noqa: BLE001 - one bad picture or report is no reason to stop watching
                log.debug("A picture or report couldn't be used", exc_info=True)

    def _one(self, item) -> None:
        if isinstance(item, VideoTrack):  # a stream has just been opened
            if self.decoder is not None:  # ... anew, after a drop
                self._reopened = True
                self.pace.reopened(time.monotonic())
            self.decoder, self.codec = Decoder(item.codec, item.parameter_sets, gpu=getattr(self.engine, "gpu_decode", True)), item.codec
            return
        if item[0] == "report":
            self._pace_state = self.pace.state(time.monotonic())  # (reports come whether or not frames do)
            for packet in self.follow.report(item[1], item[2]):
                self.emit(self._rated(packet), "auto")
            if self.follow.last_report is not None:
                self.check.report(*self.follow.last_report)
            return
        unit, at = item
        if self.decoder is None:
            return
        started = time.perf_counter()
        decoded = self.decoder.decode(unit, at)
        self.decode_ms.append((time.perf_counter() - started) * 1000)
        for rtp, frame in decoded:
            # (a frame decoded on the card comes out with the next packet: it is of its own packet's time, but has
            # waited only since it came out)
            out, at = at, self.decoder.came(rtp, at)
            if self._reopened:
                self._reopened = False
                if not self.follow.continues(rtp, at):
                    # the camera's clock started over (it was restarted): everything held of it is from
                    # another time, so the following starts afresh, under new track ids
                    stats = self.follow.stats
                    self.follow = CameraFollow(self.stamp_ms, epoch=_base36(int(time.time())))
                    self.follow.stats.update({key: stats[key] for key in ("frames", "reports", "sent")})
                    self.view.afresh()
                    self.restarts += 1
            bench = getattr(self.engine, "bench", None)
            if bench is not None:
                bench.offer(frame, self.follow.reported_moving() if bench.state == "collecting" else None)  # (bench.py)
            # is the view itself moving? Asked a few times a second, of every frame that comes (looked at or not)
            if self.view.due(rtp / 90.0):
                started = time.perf_counter()
                self.view.offer(rtp / 90.0, small_grey(frame, VIEW_W, VIEW_H))
                self.view_ms.append((time.perf_counter() - started) * 1000)
            now = time.monotonic()
            plan = self.pace.plan(max(0.0, time.time() - out / 1000.0), self.follow.busy(), now)
            self._pace_state = self.pace.state(now)
            self._floor(self._pace_state, now)
            if not plan["look"]:
                continue
            started = time.perf_counter()
            found = self.engine.look(frame, plan["far"], plan["faces"])
            (self.far_ms if plan["far"] else self.look_ms).append((time.perf_counter() - started) * 1000)
            self.follow.last_frame = None
            for packet in self.follow.frame(rtp, found, at, self.view.state):
                self.emit(self._rated(packet), "auto")  # in Protect's name: what Protect reported, refined
            for packet in self.follow.local_packets:
                self.emit(self._rated(packet), "local")  # on our own word
            if self.follow.last_frame is not None:  # the picture check: numbers only, the small picture is dropped again
                when, ours = self.follow.last_frame
                self.check.frame(when, picture(frame, CHECK_W, CHECK_H)[:, :, 1], ours)

    def _floor(self, pace: dict[str, Any], now: float) -> None:
        """Under the frame-rate floor is never passed over in silence: on
        every packet and in the status as it happens (`below`), and in the
        log once it has lasted BELOW_SAID seconds and once when it is over."""
        if pace["below"]:
            if self._below_since is None:
                self._below_since = now
            elif not self._below_said and now - self._below_since >= BELOW_SAID:
                self._below_said = True
                log.warning("A camera is under the frame-rate floor: %.1f of its frames a second are looked at, the floor is %d "
                            "(decoding takes %s ms a frame, a look %s). Frames more than a second late are let go unlooked",
                            pace["fps"], pace["floor"], _median(self.decode_ms), _median(self.look_ms))
        elif self._below_since is not None:
            if self._below_said:
                log.warning("A camera is at its frame-rate floor again (it was under it for %d s)", now - self._below_since)
            self._below_since, self._below_said = None, False

    def _rated(self, packet: dict[str, Any]) -> dict[str, Any]:
        """A packet with how the camera is being kept up with: frames looked
        at a second, the floor, and whether that floor is being held."""
        pace = self._pace_state
        packet["fps"], packet["fpsFloor"], packet["fpsOk"] = pace["fps"], pace["floor"], not pace["below"]
        packet["view"] = "moving" if self.view.state == MOVING else "steady"
        return packet

    def state(self) -> dict[str, Any]:
        looks, fars = sorted(self.look_ms), sorted(self.far_ms)
        pace = self.pace.state(time.monotonic())
        return {
            "codec": self.codec,
            # how the stream is decoded ("gpu": on the card, a large stream brought down there; "software"), its own
            # size, and what decoding a frame takes
            "decode": {"how": self.decoder.how if self.decoder else None, "size": list(self.decoder.size) if self.decoder and self.decoder.size else None,
                       "ms": _median(self.decode_ms), "gpu_error": self.decoder.gpu_error if self.decoder else None},
            **pace, "below_s": round(time.monotonic() - self._below_since) if self._below_since is not None else None, "look_ms": round(looks[len(looks) // 2], 1) if looks else None,
            "far_look_ms": round(fars[len(fars) // 2], 1) if fars else None,
            "lost": self.lost, "restarts": self.restarts, **self.follow.stats,
            "check": self.check.result(), "head_start": self.follow.head_start(), "local": dict(self.follow.local.stats),
            "face_boxes": self.follow.local.faces.stats(),
            # frame_ms: what a check costs the frame it is asked for; check_ms: how long the comparing then takes on its
            # own thread, waiting its turn included (the work itself is about 0.4); skipped: pictures it never got to
            "view": {"state": "moving" if self.view.state == MOVING else "steady", **self.view.motion.stats,
                     "moving_ms": round(self.view.motion.stats["moving_ms"]),
                     "frame_ms": _median(self.view_ms), "check_ms": _median(self.view.ms), "skipped": self.view.skipped},
        }


class Camera:
    """A camera while it is watched.

    Its stream may drop: it is opened again, after a wait that doubles each
    time up to a minute, for as long as the camera is to be watched (whether
    it is worth watching is for Home Assistant to say: a camera that has gone
    offline or been removed is told to sleep). A drop that lasts is said once
    (`stream`: "down", in the log and through `changed`), and so is its end;
    the tries in between are not. The address is never logged."""

    def __init__(self, engine, source: str, stamp_ms: float, emit: Callable[[dict[str, Any], str], None],
                 changed: Callable[[], None] | None = None) -> None:
        self.worker = CameraWorker(engine, stamp_ms, emit)
        self.worker.start()
        self.error: str | None = None
        self.opened = 0
        self.open = False  # video is coming
        self.down = False  # ... and has not been for a while
        self.retry = RETRY_FIRST  # the wait before the next try (seconds)
        self._since = time.monotonic()  # since when there is no video
        self._changed = changed
        self._task = asyncio.create_task(self._read(source))

    def report(self, packet: dict[str, Any]) -> None:
        self.worker.put(("report", packet, time.time() * 1000))

    async def stop(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self.open = False
        self.worker.stop()

    def _up(self) -> None:
        self.open, self.error = True, None
        if self.down:
            self.down = False
            log.info("A camera's stream is back (after %d s)", time.monotonic() - self._since)
            self._tell()

    def _failed(self, error: str) -> None:
        now = time.monotonic()
        if self.open:
            self.open, self._since = False, now
        self.error = error
        if not self.down and now - self._since >= DOWN_AFTER:
            self.down = True
            log.info("A camera's stream is down (%s): trying again, at most every %d s", error, RETRY_MOST)
            self._tell()

    def _tell(self) -> None:
        if self._changed is not None:
            try:
                self._changed()
            except Exception:  # noqa: BLE001
                pass

    async def _read(self, source: str) -> None:
        """Keep reading the video; start over after a drop."""
        while True:
            started = time.monotonic()
            try:
                async with RtspReader(source) as reader:
                    self.opened += 1
                    self.worker.put(reader.track)
                    depack = Depacketizer(reader.track.codec)
                    async for packet in reader.packets():
                        if not self.open:
                            self._up()
                        for unit in depack.push(packet):
                            self.worker.put((unit, time.time() * 1000))
                error = "the stream ended"
            except RtspError as err:
                error = str(err)
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - a connection cut in the middle, or anything else: tried again all the same
                error = type(err).__name__  # (never the message: it may name the address)
            if self.open and time.monotonic() - started >= RAN:
                self.retry = RETRY_FIRST
            self._failed(error)
            await asyncio.sleep(self.retry)
            self.retry = min(self.retry * 2, RETRY_MOST)

    def state(self) -> dict[str, Any]:
        stream = "open" if self.open else "down" if self.down else "opening"
        return {
            "error": self.error, "opened": self.opened, "stream": stream,
            "down_s": None if self.open else round(time.monotonic() - self._since), "retry_s": None if self.open else self.retry,
            **self.worker.state(),
        }
