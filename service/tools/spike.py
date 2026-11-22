"""Stage 0: do our own boxes line up with Protect's, and at what timing?

Stands in for the tracking service: Home Assistant's Protect Tracker
integration connects to it, tells it which camera to watch (with the stream
address) and passes Protect's live reports on. The camera's video is read
with its raw RTP timestamps and the detectors run on every frame. Afterwards
each of Protect's boxes (of things on the move) is compared with ours in the
frames round it, for a range of "our frame's stamp minus the report's", and
the timing where they overlap best is printed, by kind.

    python -m tools.spike --models <folder> --seconds 600 --save run.json

Nothing is written to Protect; no address is printed or saved. --save keeps
the boxes and timings (no pictures), so --load can analyse them again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

from aiohttp import WSMsgType, web
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracker_service.clock import RTP_WRAP_MS  # noqa: E402
from tracker_service.detect import Detector, FaceDetector, find_faces, iou  # noqa: E402
from tracker_service.rtp import Depacketizer, Unwrap  # noqa: E402
from tracker_service.rtsp import RtspError, RtspReader, VideoTrack  # noqa: E402
from tracker_service.video import Decoder  # noqa: E402

STEPS = list(range(-800, 601, 25))  # ms tried: our frame's stamp minus the report's


def now_ms() -> float:
    return time.time() * 1000


class Looker(threading.Thread):
    """Decodes and detects, off the event loop. When it falls behind, the
    pictures that don't fit in the queue are dropped (and counted)."""

    def __init__(self, models: Path, model: str, faces: bool) -> None:
        super().__init__(daemon=True)
        self.units: queue.Queue = queue.Queue(maxsize=120)
        self.frames: list[dict] = []  # {rtp, at (ours, ms, when its last packet came), done, boxes: {kind: [..]}}
        self.dropped = 0
        self.times = {"decode": [], "detect": [], "face": []}
        self.size = None
        self.codec = None
        self.detector = Detector(models / f"yolox_{model}.onnx")
        face_model = models / "face_yunet.onnx"
        self.face = FaceDetector(face_model) if faces and face_model.exists() else None
        self.decoder = None

    def run(self) -> None:
        while True:
            item = self.units.get()
            if item is None:
                return
            if isinstance(item, VideoTrack):  # a stream has just been opened
                self.decoder, self.codec = Decoder(item.codec, item.parameter_sets), item.codec
                continue
            unit, at = item
            t0 = time.perf_counter()
            decoded = self.decoder.decode(unit)
            t1 = time.perf_counter()
            for rtp, picture in decoded:
                self.size = picture.shape[1::-1]
                self.times["decode"].append(t1 - t0)
                t2 = time.perf_counter()
                boxes = self.detector.detect(picture, 0.3, ("person", "vehicle", "animal"))
                t3 = time.perf_counter()
                self.times["detect"].append(t3 - t2)
                if self.face is not None and boxes["person"]:
                    boxes["face"] = find_faces(self.face, picture, boxes["person"], 0.6)
                    self.times["face"].append(time.perf_counter() - t3)
                self.frames.append({"rtp": rtp, "at": at, "done": now_ms(), "boxes": boxes})


class Camera:
    """What was gathered for one camera: Protect's reports, and our frames."""

    def __init__(self, args) -> None:
        self.looker = Looker(Path(args.models), args.model, not args.no_faces)
        self.looker.start()
        self.reports: list[dict] = []
        self.task: asyncio.Task | None = None
        self.opened = 0
        self.channel = None
        self.error = None

    async def read(self, source: str) -> None:
        depack = None
        try:
            async with RtspReader(source) as reader:
                self.opened += 1
                self.looker.units.put(reader.track)
                depack = Depacketizer(reader.track.codec)
                async for packet in reader.packets():
                    for unit in depack.push(packet):
                        try:
                            self.looker.units.put_nowait((unit, now_ms()))
                        except queue.Full:
                            self.looker.dropped += 1
        except RtspError as err:
            self.error = str(err)
            print(f"  video: {err}", flush=True)

    def data(self, args) -> dict:
        look = self.looker
        return {
            "codec": look.codec, "size": look.size, "channel": self.channel, "model": args.model, "device": look.detector.device,
            "dropped": look.dropped, "opened": self.opened, "frames": list(look.frames), "reports": list(self.reports),
            "times": {k: [round(v * 1000, 2) for v in vals] for k, vals in look.times.items()},
        }


async def watch(args) -> dict:
    """Wait for Home Assistant's Protect Tracker integration to connect, and
    gather what it sends and what the cameras show, for --seconds from the
    first camera it tells us to watch."""
    cameras: dict[str, Camera] = {}
    started: list[float] = []

    async def link(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
        await ws.prepare(request)
        print("Home Assistant connected", flush=True)
        await ws.send_json({"type": "status", "mode": "measuring"})
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                break
            data = json.loads(msg.data)
            cam = data.get("camera")
            if data.get("type") == "watch":
                camera = cameras.get(cam) or cameras.setdefault(cam, Camera(args))
                camera.channel = data.get("channel")
                if camera.task is None or camera.task.done():
                    camera.task = asyncio.create_task(camera.read(data["source"]))
                    started or started.append(time.monotonic())
                    print(f"  watching camera {list(cameras).index(cam)} (stream {camera.channel})", flush=True)
            elif data.get("type") == "sleep" and cam in cameras:
                if cameras[cam].task:
                    cameras[cam].task.cancel()
                print(f"  camera {list(cameras).index(cam)} sleeps", flush=True)
            elif data.get("type") == "report" and cam in cameras:
                packet = data.get("packet") or {}
                if packet.get("clockStream") is not None:
                    packet["got"] = now_ms()
                    cameras[cam].reports.append(packet)
        print("Home Assistant disconnected", flush=True)
        return ws

    app = web.Application()
    app.router.add_get("/link", link)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    print(f"waiting for Home Assistant on {args.host}:{args.port}", flush=True)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopping.set)
    said = time.monotonic()
    while not stopping.is_set() and not (started and time.monotonic() - started[0] > args.seconds):
        try:
            await asyncio.wait_for(stopping.wait(), 1)
        except asyncio.TimeoutError:
            pass
        if time.monotonic() - said > 60:
            said = time.monotonic()
            for i, camera in enumerate(cameras.values()):
                moving = sum(1 for r in camera.reports for t in r["tracks"] if not t.get("stationary"))
                print(f"  camera {i}: {len(camera.looker.frames)} frames, {len(camera.reports)} reports ({moving} moving boxes), {camera.looker.dropped} dropped", flush=True)
            if args.save:
                Path(args.save).write_text(json.dumps({"cameras": [c.data(args) for c in cameras.values()]}))
    for camera in cameras.values():
        if camera.task:
            camera.task.cancel()
        camera.looker.units.put(None)
        camera.looker.join(timeout=10)
    await runner.cleanup()
    return {"cameras": [c.data(args) for c in cameras.values()]}


def stream_ms(report: dict) -> float:
    return float(report["clockStream"]) * 1000.0 / float(report.get("clockStreamRate") or 1000)


def analyse(data: dict, out=print) -> dict:
    frames, reports = data["frames"], data["reports"]
    result: dict = {"frames": len(frames), "reports": len(reports)}
    if len(frames) < 2:
        out("no frames")
        return result
    count_on = Unwrap()  # the 32-bit timestamps may start over during the run
    rtp = np.array([count_on(f["rtp"]) for f in frames], dtype=np.float64) * 1000.0 / 90000.0
    at = np.array([f["at"] for f in frames])
    span = (at[-1] - at[0]) / 1000
    result["fps"] = round((len(frames) - 1) / span, 2) if span > 0 else None
    out(f"frames: {len(frames)} in {span:.0f} s = {result['fps']} a second, {data.get('size')}, {data.get('dropped')} dropped")
    for key, vals in (data.get("times") or {}).items():
        if vals:
            out(f"  {key}: median {np.median(vals):.1f} ms, 95th {np.percentile(vals, 95):.1f} ms ({len(vals)})")
    if not reports:
        out("no reports from Protect in that time: nothing to compare")
        return result

    # every frame's time on the stream clock the reports use (the round of the 13-hour clock nearest the reports)
    ref = stream_ms(reports[0])
    frame_ms = rtp + np.round((ref - rtp[0]) / RTP_WRAP_MS) * RTP_WRAP_MS
    lag = ref - frame_ms[np.argmin(np.abs(at - reports[0]["got"]))]
    result["clock_gap_ms"] = round(float(lag))
    out(f"clocks: the frame arriving with the first report is {lag:.0f} ms older than that report's stamp"
        f"{'' if -2000 < lag < 10000 else '  <-- NOT the same clock'}")
    gaps = np.diff([stream_ms(r) for r in reports])
    gaps = gaps[(gaps > 0) & (gaps < 2000)]
    if len(gaps):
        out(f"reports: {len(reports)}, typically {np.median(gaps):.0f} ms apart ({1000 / np.median(gaps):.1f} a second)")
    kinds_seen: dict[str, int] = {}
    for r in reports:
        for t in r["tracks"]:
            kinds_seen[t.get("objectType") or "?"] = kinds_seen.get(t.get("objectType") or "?", 0) + 1
    out(f"Protect's boxes by kind: {kinds_seen}")

    # things on the move: a track's box against its box one report before
    last: dict[str, tuple] = {}
    moving = []  # (stream ms, kind, box x1y1x2y2 0-1, when the report came)
    for r in reports:
        when = stream_ms(r)
        for t in r["tracks"]:
            x, y, w, h = (float(v) / 1000 for v in t["coord"])
            box = (x, y, x + w, y + h)
            before = last.get(t["id"])
            last[t["id"]] = (when, box)
            if before is None or when - before[0] > 600:
                continue
            shift = abs((box[0] + box[2]) - (before[1][0] + before[1][2])) / 2 + abs((box[1] + box[3]) - (before[1][1] + before[1][3])) / 2
            if shift >= args_min_shift(box):
                moving.append((when, t.get("objectType") or "?", box, r["got"]))
    out(f"boxes of things on the move: {len(moving)}")

    by_kind: dict[str, dict] = {}
    for when, kind, box, got in moving:
        ours_kind = kind if kind in ("person", "vehicle", "animal", "face") else None
        if ours_kind is None:
            continue
        acc = by_kind.setdefault(kind, {"sum": np.zeros(len(STEPS)), "n": np.zeros(len(STEPS)), "early": []})
        for k, step in enumerate(STEPS):
            i = int(np.argmin(np.abs(frame_ms - (when + step))))
            if abs(frame_ms[i] - (when + step)) > 25:
                continue
            best = max((iou(box, b) for b in frames[i]["boxes"].get(ours_kind) or []), default=0.0)
            acc["sum"][k] += best
            acc["n"][k] += 1
        acc["early"].append((when, got))
    result["kinds"] = {}
    for kind, acc in by_kind.items():
        ok = acc["n"] >= max(5, acc["n"].max() * 0.5)
        if not ok.any():
            out(f"{kind}: too few to tell")
            continue
        mean = np.where(ok, acc["sum"] / np.maximum(acc["n"], 1), 0)
        k = int(mean.argmax())
        good = [STEPS[j] for j in range(len(STEPS)) if mean[j] >= mean[k] * 0.97]
        # by a parabola through the top three, for a figure finer than the steps
        fine = float(STEPS[k])
        if 0 < k < len(STEPS) - 1 and ok[k - 1] and ok[k + 1]:
            a, b, c = mean[k - 1], mean[k], mean[k + 1]
            if a - 2 * b + c < 0:
                fine += 12.5 * (a - c) / (a - 2 * b + c)
        at_step = {s: round(float(mean[STEPS.index(s)]), 3) for s in (-125, -25, 0) if ok[STEPS.index(s)]}
        # how long before Protect's report came was our frame for that moment done
        done = np.array([f["done"] for f in frames])
        early = []
        for when, got in acc["early"]:
            i = int(np.argmin(np.abs(frame_ms - (when + fine))))
            if abs(frame_ms[i] - (when + fine)) <= 25:
                early.append(got - done[i])
        result["kinds"][kind] = {"n": int(acc["n"].max()), "best_ms": round(fine), "overlap": round(float(mean[k]), 3), "range": [good[0], good[-1]], "at": at_step,
                                 "ours_ahead_ms": round(float(np.median(early))) if early else None}
        out(f"{kind}: {int(acc['n'].max())} boxes; overlap best {mean[k]:.2f} when our frame is {fine:+.0f} ms from the report's stamp "
            f"(within 3%: {good[0]:+d} .. {good[-1]:+d}); overlap at -125/-25/0: {at_step}; "
            f"our box for that frame was ready {np.median(early):.0f} ms before Protect's report came" if early else f"{kind}: no frames matched")
    return result


def args_min_shift(box) -> float:
    """A box counts as on the move when its middle went at least this far
    since the report before: a tenth of its own smaller side, and never less
    than half a percent of the picture."""
    return max(0.005, 0.1 * min(box[2] - box[0], box[3] - box[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--models", default=os.environ.get("TRACKER_MODELS", "models"))
    ap.add_argument("--model", default="m")
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--load", help="analyse a file kept with --save")
    args = ap.parse_args()
    if args.load:
        data = json.loads(Path(args.load).read_text())
    else:
        data = asyncio.run(watch(args))
        if args.save:
            Path(args.save).write_text(json.dumps(data))
    for i, camera in enumerate(data["cameras"]):
        print(f"== camera {i}, stream {camera.get('channel')}, {camera.get('codec')}, yolox_{camera.get('model')} on {camera.get('device')}, video opened {camera.get('opened')} times")
        analyse(camera)


if __name__ == "__main__":
    main()
