"""Stage 0: do our own boxes line up with Protect's, and at what timing?

Reads one camera's video with its raw RTP timestamps, runs the detectors on
every frame, and listens to Protect's live reports for the same camera through
Home Assistant. Afterwards each of Protect's boxes (of things on the move) is
compared with ours in the frames round it, for a range of "the frame is this
much earlier than the report's stamp", and the timing where they overlap best
is printed, by kind.

    HA_URL=... HA_TOKEN=... python -m tools.spike --seconds 600

(or --env-file a file holding those two lines). The camera is chosen by
--camera (a number from the list it prints, or an entity id); the video is
taken from Home Assistant's own stream relay by default (--video for another
RTSP address). Nothing is written to Protect, and no address is printed.

--save keeps the boxes and timings (no pictures) so analyse() can be run again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracker_service.clock import RTP_WRAP_MS  # noqa: E402
from tracker_service.detect import Detector, FaceDetector, find_faces, iou  # noqa: E402
from tracker_service.rtp import Depacketizer  # noqa: E402
from tracker_service.rtsp import RtspReader  # noqa: E402
from tracker_service.video import Decoder  # noqa: E402

STEPS = list(range(-800, 601, 25))  # ms tried: our frame's stamp minus the report's


def now_ms() -> float:
    return time.time() * 1000


def read_env(path: str | None) -> tuple[str, str]:
    env = dict(os.environ)
    if path:
        for line in Path(path).read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                env[key.strip()] = value.strip().strip("'\"")
    url, token = env.get("HA_URL"), env.get("HA_TOKEN")
    if not url or not token:
        raise SystemExit("HA_URL and HA_TOKEN are needed (environment or --env-file)")
    return url.rstrip("/"), token


class Ha:
    """Just enough of Home Assistant's websocket."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.n = 0

    @classmethod
    async def connect(cls, session, url: str, token: str) -> Ha:
        ws = await session.ws_connect(url.replace("http", "ws", 1) + "/api/websocket", max_msg_size=0)
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        if (await ws.receive_json()).get("type") != "auth_ok":
            raise SystemExit("Home Assistant refused the token")
        return cls(ws)

    async def call(self, **msg):
        self.n += 1
        await self.ws.send_json({"id": self.n, **msg})
        while True:
            reply = await self.ws.receive_json()
            if reply.get("id") == self.n and reply.get("type") == "result":
                if not reply.get("success"):
                    raise SystemExit(f"{msg['type']}: {(reply.get('error') or {}).get('message')}")
                return reply.get("result")


OFFER = "\r\n".join([
    "v=0", "o=- 4611731400430051336 2 IN IP4 127.0.0.1", "s=-", "t=0 0", "a=group:BUNDLE 0", "a=msid-semantic: WMS",
    "m=video 9 UDP/TLS/RTP/SAVPF 96 97", "c=IN IP4 0.0.0.0", "a=rtcp:9 IN IP4 0.0.0.0", "a=ice-ufrag:spke",
    "a=ice-pwd:spikespikespikespikespike", "a=ice-options:trickle",
    "a=fingerprint:sha-256 " + ":".join(["AB"] * 32), "a=setup:actpass", "a=mid:0", "a=recvonly", "a=rtcp-mux",
    "a=rtpmap:96 H264/90000", "a=fmtp:96 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f",
    "a=rtpmap:97 H265/90000", "",
])


async def wake_relay(ha: Ha, entity: str) -> None:
    """Home Assistant's stream relay only knows a camera once a live view of
    it has been asked for. This asks, as a browser would, and never connects:
    the relay then has the stream under the entity's id."""
    ha.n += 1
    mine = ha.n
    await ha.ws.send_json({"id": mine, "type": "camera/webrtc/offer", "entity_id": entity, "offer": OFFER})
    end = time.monotonic() + 15
    while time.monotonic() < end:
        try:
            reply = await asyncio.wait_for(ha.ws.receive_json(), 15)
        except asyncio.TimeoutError:
            break
        if reply.get("id") != mine:
            continue
        if reply.get("type") == "result" and not reply.get("success"):
            raise SystemExit(f"live view refused: {(reply.get('error') or {}).get('message')}")
        kind = (reply.get("event") or {}).get("type")
        if kind in ("answer", "error"):
            break
    await ha.call(type="unsubscribe_events", subscription=mine)


class Looker(threading.Thread):
    """Decodes and detects, off the event loop. Newest pictures win: when it
    falls behind, the oldest waiting ones are dropped (and counted)."""

    def __init__(self, models: Path, model: str, track, faces: bool) -> None:
        super().__init__(daemon=True)
        self.units: queue.Queue = queue.Queue(maxsize=120)
        self.frames: list[dict] = []  # {rtp, at (ours, ms, when its last packet came), boxes: {kind: [..]}}
        self.dropped = 0
        self.times = {"decode": [], "detect": [], "face": []}
        self.size = None
        self.detector = Detector(models / f"yolox_{model}.onnx")
        face_model = models / "face_yunet.onnx"
        self.face = FaceDetector(face_model) if faces and face_model.exists() else None
        self.decoder = Decoder(track.codec, track.parameter_sets)

    def run(self) -> None:
        while True:
            item = self.units.get()
            if item is None:
                return
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


async def watch(args) -> dict:
    url, token = read_env(args.env_file)
    async with aiohttp.ClientSession() as session:
        ha = await Ha.connect(session, url, token)
        cameras = sorted(((await ha.call(type="protect_detections/live/cameras")) or {}).get("cameras") or {})
        if not cameras:
            raise SystemExit("No camera with live tracking")
        if args.camera is None or (args.camera.isdigit() and int(args.camera) >= len(cameras)):
            for i, entity in enumerate(cameras):
                print(f"  {i}: {entity}")
            raise SystemExit("Pick one with --camera")
        entity = cameras[int(args.camera)] if args.camera.isdigit() else args.camera
        video = args.video or f"rtsp://127.0.0.1:18554/{entity}"
        if not args.video:
            await wake_relay(ha, entity)

        reports: list[dict] = []
        async with RtspReader(video) as reader:
            track = reader.track
            looker = Looker(Path(args.models), args.model, track, not args.no_faces)
            looker.start()
            print(f"video: {track.codec}; detector yolox_{args.model} on {looker.detector.device}"
                  f"{' (fast copy)' if looker.detector.fast else ''}; faces {'on' if looker.face else 'off'}", flush=True)
            await ha.call(type="protect_detections/live/subscribe", entity=entity)
            stop = time.monotonic() + args.seconds

            async def feed() -> None:
                async for msg in ha.ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        break
                    event = json.loads(msg.data).get("event")
                    if isinstance(event, dict) and event.get("clockStream") is not None:
                        event["got"] = now_ms()
                        reports.append(event)

            feeding = asyncio.create_task(feed())
            depack = Depacketizer(track.codec)
            said = time.monotonic()
            try:
                async for packet in reader.packets():
                    for unit in depack.push(packet):
                        try:
                            looker.units.put_nowait((unit, now_ms()))
                        except queue.Full:
                            looker.dropped += 1
                    if time.monotonic() > stop:
                        break
                    if time.monotonic() - said > 30:
                        said = time.monotonic()
                        moving = sum(1 for r in reports for t in r["tracks"] if not t.get("stationary"))
                        print(f"  {len(looker.frames)} frames, {len(reports)} reports ({moving} moving boxes), {looker.dropped} dropped", flush=True)
            finally:
                feeding.cancel()
                looker.units.put(None)
                looker.join(timeout=10)
    return {
        "codec": track.codec, "size": looker.size, "model": args.model, "device": looker.detector.device,
        "dropped": looker.dropped, "frames": looker.frames, "reports": reports,
        "times": {k: [round(v * 1000, 2) for v in vals] for k, vals in looker.times.items()},
    }


def stream_ms(report: dict) -> float:
    return float(report["clockStream"]) * 1000.0 / float(report.get("clockStreamRate") or 1000)


def analyse(data: dict, out=print) -> dict:
    frames, reports = data["frames"], data["reports"]
    result: dict = {"frames": len(frames), "reports": len(reports)}
    if len(frames) < 2:
        out("no frames")
        return result
    rtp = np.array([f["rtp"] for f in frames], dtype=np.float64) * 1000.0 / 90000.0
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
    ap.add_argument("--env-file")
    ap.add_argument("--camera")
    ap.add_argument("--video", help="an RTSP address instead of Home Assistant's relay")
    ap.add_argument("--models", default=os.environ.get("TRACKER_MODELS", "models"))
    ap.add_argument("--model", default="m")
    ap.add_argument("--seconds", type=int, default=300)
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
    analyse(data)


if __name__ == "__main__":
    main()
