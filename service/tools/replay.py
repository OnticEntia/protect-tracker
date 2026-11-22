"""Replay: a stretch of a camera's recording put through the detector and
tracker as if it were live, to see what the tracker makes of it.

The recording is the console's own: it is asked for through Home Assistant
(the UniFi Protect integration's video proxy), held in memory, decoded there,
and dropped. Nothing is written but the numbers printed; no address is
printed or kept.

    HA_URL=... HA_TOKEN=... python -m tools.replay --models <folder> \\
        --start 2026-01-31T21:17:30Z --end 2026-01-31T21:18:30Z [--camera 0] [--old]

For each person the tracker followed: from when to when, and for each time
one track ended and another began, what was going on (hidden: nobody found
there in between; shape: found, but much taller or shorter than before;
border: at the picture's edge; beside: two tracks at once). `--old` runs the
tracker as it was before people were held on to through being hidden, for
comparison. `--both` runs both.

What it cannot replay: Protect's own live reports (the console keeps no
history of them), so this is the tracker on its own word, as in Local mode.
The recording is the camera's main stream scaled down, not the very stream
the live service watches, so a box may differ by a little.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracker_service import local as local_rules  # noqa: E402
from tracker_service import tracks as track_rules  # noqa: E402
from tracker_service.follow import CameraFollow  # noqa: E402

LIMIT = 600 * 1024 * 1024  # the most recording held in memory at once


@contextmanager
def old_rules():
    """The tracker as it was before people were held on to: dropped after half
    a second unseen, not known again further off, nested boxes both kept."""
    saved = (track_rules.GONE_PERSON_MS, track_rules.INSIDE, track_rules.REACH_PER_S, local_rules.SHOWN_UNSEEN_MS)
    track_rules.GONE_PERSON_MS, track_rules.INSIDE, track_rules.REACH_PER_S, local_rules.SHOWN_UNSEEN_MS = track_rules.GONE_MS, 2.0, 0.0, 500
    try:
        yield
    finally:
        track_rules.GONE_PERSON_MS, track_rules.INSIDE, track_rules.REACH_PER_S, local_rules.SHOWN_UNSEEN_MS = saved


def _box(coord) -> tuple:
    x, y, w, h = (v / 1000.0 for v in coord)
    return (x, y, x + w, y + h)


def _near(a, b, widths: float = 2.0) -> bool:
    return abs((a[0] + a[2]) - (b[0] + b[2])) / 2 <= widths * max(a[2] - a[0], b[2] - b[0]) and abs((a[1] + a[3]) - (b[1] + b[3])) / 2 <= max(a[3] - a[1], b[3] - b[1])


def replay(looks, old: bool = False) -> dict[str, Any]:
    """`looks`: (time in ms, {kind: [(x1, y1, x2, y2, score)]}) per frame, in
    order. Returns what was told of people, and where their tracks broke."""
    with old_rules() if old else _as_is():
        follow = CameraFollow(epoch="r")
        told: dict[str, dict[str, Any]] = {}  # track id -> first, last, boxes at both ends
        faces: dict[str, int] = {}
        seen: list[tuple[float, list]] = []  # per frame: the people the detector found
        frames = most = 0
        for ms, found in looks:
            frames += 1
            seen.append((ms, [b[:4] for b in found.get("person", [])]))
            follow.frame(int(ms * 90) % (1 << 32), found, ms)
            for packet in follow.local_packets:
                people = [t for t in packet["tracks"] if t["objectType"] == "person"]
                most = max(most, len(people))
                for t in packet["tracks"]:
                    if t["objectType"] == "face":
                        faces[t["id"]] = faces.get(t["id"], 0) + 1
                for t in people:
                    row = told.setdefault(t["id"], {"id": t["id"], "first": ms, "first_box": _box(t["coord"]), "frames": 0})
                    row["last"], row["last_box"], row["frames"] = ms, _box(t["coord"]), row["frames"] + 1
        face_stats = follow.local.faces.stats()
    rows = sorted(told.values(), key=lambda r: r["first"])
    breaks = []
    for i, row in enumerate(rows[1:], 1):
        # the track this one most likely carries on from: the one that ended latest before it began, nearby
        before = [r for r in rows[:i] if _near(r["last_box"], row["first_box"], 4.0)]
        if not before:
            continue
        prev = max(before, key=lambda r: min(r["last"], row["first"]))
        gap = row["first"] - prev["last"]
        # (what matters is the moment the first track was lost: was anyone to be found there just then?)
        between = [boxes for ms, boxes in seen if prev["last"] < ms < min(row["first"], prev["last"] + 700)]
        found_near = sum(1 for boxes in between if any(_near(b, prev["last_box"], 3.0) for b in boxes))
        tall_before, tall_after = prev["last_box"][3] - prev["last_box"][1], row["first_box"][3] - row["first_box"][1]
        edge = min(prev["last_box"][0], prev["last_box"][1], 1 - prev["last_box"][2], 1 - prev["last_box"][3]) <= 0.05
        if gap <= 0:
            why = "beside: both at once"
        elif edge:
            why = "border: the first was at the picture's edge"
        elif between and found_near < 0.3 * len(between):
            why = "hidden: nobody found there in between"
        elif not 0.6 <= tall_after / max(tall_before, 1e-6) <= 1.6:
            why = "shape: found again much taller or shorter"
        else:
            why = "found there throughout, and not taken as the same"
        breaks.append({"at": row["first"], "gap_ms": round(gap), "why": why, "from": prev["id"], "to": row["id"]})
    return {
        "frames": frames,
        "people_frames": sum(1 for _, boxes in seen if boxes),
        "tracks": [{"id": r["id"], "from": r["first"], "to": r["last"], "seconds": round((r["last"] - r["first"]) / 1000, 1), "frames": r["frames"]} for r in rows],
        "most_at_once": most,
        "breaks": breaks,
        "face_tracks": len(faces),
        "face_frames": sum(faces.values()),
        "face_boxes": face_stats,
    }


@contextmanager
def _as_is():
    yield


# ------------------------------------------------------------ the recording

async def fetch(url: str, token: str, camera: str | None, start: datetime, end: datetime) -> bytes:
    """The recording between two times, through Home Assistant, into memory."""
    import aiohttp  # noqa: PLC0415

    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(url.replace("http", "ws", 1) + "/api/websocket", max_msg_size=0)
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        if (await ws.receive_json()).get("type") != "auth_ok":
            raise SystemExit("Home Assistant refused the token")

        async def call(n: int, **msg):
            await ws.send_json({"id": n, **msg})
            while True:
                reply = await ws.receive_json()
                if reply.get("id") == n:
                    if not reply.get("success"):
                        raise SystemExit(f"{msg['type']}: {(reply.get('error') or {}).get('message')}")
                    return reply.get("result")

        entries = await call(1, type="config_entries/get", domain="unifiprotect")
        if not entries:
            raise SystemExit("The UniFi Protect integration isn't set up")
        cameras = ((await call(2, type="protect_detections/live/cameras")) or {}).get("cameras") or {}
        names = sorted(cameras)
        if camera is None or (camera.isdigit() and int(camera) >= len(names)) or (not camera.isdigit() and camera not in cameras):
            for i, name in enumerate(names):
                print(f"  {i}: {name}")
            raise SystemExit("Pick one with --camera")
        camera_id = cameras[names[int(camera)] if camera.isdigit() else camera]
        await ws.close()
        path = f"/api/unifiprotect/video/{entries[0]['entry_id']}/{camera_id}/{quote(start.isoformat())}/{quote(end.isoformat())}"
        async with session.get(url + path, headers={"Authorization": f"Bearer {token}"}) as resp:
            if resp.status != 200:
                raise SystemExit(f"The recording could not be had ({resp.status})")
            data = bytearray()
            async for chunk in resp.content.iter_chunked(1 << 20):
                data += chunk
                if len(data) > LIMIT:
                    raise SystemExit("That is more recording than is held in memory at once: ask for a shorter stretch")
            return bytes(data)


def looks_of(recording: bytes, engine, sharp: bool = True, every: int = 1):
    """The recording decoded in memory, each frame looked at: (ms, found)."""
    import av  # noqa: PLC0415

    with av.open(io.BytesIO(recording)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i % every or frame.pts is None:
                continue
            yield float(frame.pts * stream.time_base) * 1000.0, engine.look(frame, far=sharp, faces=True)


def when(text: str) -> datetime:
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def say(result: dict[str, Any], label: str) -> None:
    clock = lambda ms: f"{ms / 1000:6.1f}s"  # noqa: E731
    print(f"-- {label}: {result['frames']} frames, a person found in {result['people_frames']}; "
          f"{len(result['tracks'])} person track(s), {result['most_at_once']} at once at most; "
          f"faces in {result['face_frames']} frames ({result['face_tracks']} face track(s))")
    f = result["face_boxes"]
    if f["found"]:
        print(f"   face boxes: {f['found']} from a find in that frame, {f['carried']} carried along with the person; at a find the carried box was "
              f"{f['carried_off']} of a face's width off, one left where it was last found would have been {f['held_off']} ({f['compared']} compared)")
    for t in result["tracks"]:
        print(f"   {t['id']:>8}  {clock(t['from'])} to {clock(t['to'])}  ({t['seconds']} s, {t['frames']} frames)")
    for b in result["breaks"]:
        print(f"   break at {clock(b['at'])}: {b['from']} -> {b['to']}, {b['gap_ms']} ms apart: {b['why']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", help="a file with HA_URL= and HA_TOKEN= lines (else the environment)")
    ap.add_argument("--camera")
    ap.add_argument("--start", required=True, type=when)
    ap.add_argument("--end", required=True, type=when)
    ap.add_argument("--models", default=os.environ.get("TRACKER_MODELS", "models"))
    ap.add_argument("--model", default="m")
    ap.add_argument("--ordinary", action="store_true", help="the ordinary look, not the sharp one")
    ap.add_argument("--old", action="store_true", help="the tracker as it was before people were held on to")
    ap.add_argument("--both", action="store_true", help="the old rules and the present ones, one after the other")
    args = ap.parse_args()
    env = dict(os.environ)
    if args.env_file:
        for line in Path(args.env_file).read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                env[key.strip()] = value.strip().strip("'\"")
    if not env.get("HA_URL") or not env.get("HA_TOKEN"):
        raise SystemExit("HA_URL and HA_TOKEN are needed (environment or --env-file)")
    from tracker_service.camera import Engine  # noqa: PLC0415

    recording = asyncio.run(fetch(env["HA_URL"].rstrip("/"), env["HA_TOKEN"], args.camera, args.start, args.end))
    engine = Engine(Path(args.models), args.model)
    looks = list(looks_of(recording, engine, sharp=not args.ordinary))  # (boxes only from here on: the pictures are gone)
    del recording
    print(f"{args.start.isoformat()} to {args.end.isoformat()}: {len(looks)} frames, the {'ordinary' if args.ordinary else 'sharp'} look, yolox_{args.model}")
    for old in ([True, False] if args.both else [args.old]):
        say(replay(looks, old=old), "old rules" if old else "present rules")


if __name__ == "__main__":
    main()
