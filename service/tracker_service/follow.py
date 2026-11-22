"""One camera: Protect's tracks, followed in our own frames.

Protect is the authority. It says what is there (its track ids, kinds,
descriptions, zones, event); the detector only refines where. So nothing is
sent for a thing Protect has not reported, and everything that is sent carries
Protect's id and details. What this adds is a box for every frame instead of
four a second.

How a local track is tied to one of Protect's: a report with clock T
describes the frame stamped T + stamp (stamp is negative and measured, about
an eighth of a second). Our own tracks' boxes of that frame are compared with
the report's; the one that overlaps a Protect track best is bound to it, and
from then on its box in each new frame is sent under Protect's id. When the
two stop agreeing for a few reports in a row the binding is dropped, and the
cards are back to Protect's own boxes until one fits again. The same when the
local track is lost. Things Protect calls stationary are left to Protect: its
box of a parked car is steadier than a fresh guess thirty times a second.

Times here are the camera's wall clock (ms), and every packet that goes out
is stamped with the time of the frame its boxes belong to (`stampMs: 0`).

Plain Python: no video, no model, no network. Fed by camera.py.
"""

from __future__ import annotations

from typing import Any

from .clock import FeedClock
from .detect import iou
from .tracks import LocalTracker, Track

KINDS = {"person": "person", "vehicle": "vehicle", "animal": "animal", "face": "face"}  # Protect's kind -> ours
BIND_IOU = 0.4  # a local track is bound to a Protect track when they overlap this much in the report's frame
KEEP_IOU = 0.25  # ... and stays bound while they overlap at least this much
MISSES = 3  # reports in a row that disagree before a binding is dropped
COAST_MS = 300  # a bound track not found in a frame keeps moving as it did for this long
END_MS = 1500  # no report of a Protect track for this long (by frame time): it has ended
META = ("objectType", "confidence", "zones", "loiterZones", "lines", "desc")


def to_box(coord) -> tuple:
    x, y, w, h = (float(v) / 1000.0 for v in coord)
    return (x, y, x + w, y + h)


def to_coord(box) -> list:
    x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in box)
    return [round(x1 * 1000, 1), round(y1 * 1000, 1), round((x2 - x1) * 1000, 1), round((y2 - y1) * 1000, 1)]


class _Theirs:
    """One of Protect's tracks, as last reported."""

    def __init__(self, track_id: str) -> None:
        self.id = track_id
        self.meta: dict[str, Any] = {}
        self.kind: str | None = None
        self.stationary = False
        self.reported = 0.0  # the frame time of its last report
        self.local: int | None = None  # the local track bound to it
        self.misses = 0
        self.sent = 0.0  # the newest frame time a local box of it went out for


class CameraFollow:
    def __init__(self, stamp_ms: float = -125.0) -> None:
        self.stamp = float(stamp_ms)
        self.clock = FeedClock()
        self.tracker = LocalTracker()
        self.theirs: dict[str, _Theirs] = {}
        self.event: dict[str, Any] = {}
        self.stats = {"frames": 0, "untimed": 0, "reports": 0, "bound": 0, "dropped": 0, "sent": 0}

    # ------------------------------------------------------------- reports

    def report(self, packet: dict[str, Any], now_ms: float) -> list[dict[str, Any]]:
        """One of Protect's live reports. Returns packets to send (the boxes
        already found of a track that has just been bound)."""
        wall, stream = packet.get("clockWall"), packet.get("clockStream")
        if not isinstance(wall, (int, float)) or not isinstance(stream, (int, float)):
            return []
        self.clock.report(wall, stream, packet.get("clockStreamRate"), now_ms)
        self.stats["reports"] += 1
        frame_time = wall + self.stamp  # the frame these boxes belong to
        self.event = {"eventId": packet.get("eventId"), "edgeType": packet.get("edgeType")}
        boxes: dict[str, tuple] = {}
        for tr in packet.get("tracks") or []:
            if not isinstance(tr, dict) or not isinstance(tr.get("coord"), list) or len(tr["coord"]) != 4:
                continue
            tid = str(tr.get("id"))
            mine = self.theirs.get(tid) or self.theirs.setdefault(tid, _Theirs(tid))
            mine.meta = {key: tr.get(key) for key in META}
            mine.kind = KINDS.get(tr.get("objectType"))
            mine.stationary = bool(tr.get("stationary"))
            mine.reported = frame_time
            boxes[tid] = to_box(tr["coord"])
        for tid in [tid for tid, t in self.theirs.items() if frame_time - t.reported > END_MS]:
            del self.theirs[tid]

        # bindings that still hold, and those that don't
        for tid, box in boxes.items():
            mine = self.theirs[tid]
            if mine.local is None:
                continue
            local = self.tracker.tracks.get(mine.local)
            ours = local.at(frame_time) if local is not None else None
            if mine.stationary or local is None:
                self._unbind(mine)
            elif ours is not None and iou(ours, box) >= KEEP_IOU:
                mine.misses = 0
            else:
                mine.misses += 1
                if mine.misses >= MISSES:
                    self._unbind(mine)

        # new bindings: the best overlaps first, each local track for one of theirs
        taken = {t.local for t in self.theirs.values() if t.local is not None}
        pairs = []
        for tid, box in boxes.items():
            mine = self.theirs[tid]
            if mine.local is not None or mine.kind is None or mine.stationary:
                continue
            for local in self.tracker.tracks.values():
                if local.kind != mine.kind or local.id in taken:
                    continue
                ours = local.at(frame_time)
                if ours is not None:
                    overlap = iou(ours, box)
                    if overlap >= BIND_IOU:
                        pairs.append((overlap, tid, local.id))
        out: list[dict[str, Any]] = []
        for _, tid, lid in sorted(pairs, reverse=True):
            mine = self.theirs[tid]
            if mine.local is not None or lid in taken:
                continue
            mine.local, mine.misses = lid, 0
            taken.add(lid)
            self.stats["bound"] += 1
            # what was found of it since that frame goes out now, in order
            start = max(frame_time, mine.sent)
            for when, box in self.tracker.tracks[lid].history:
                if when > start:
                    out.append(self._packet(when, [self._entry(mine, box)]))
                    mine.sent = when
        self.stats["sent"] += len(out)
        return out

    def _unbind(self, mine: _Theirs) -> None:
        mine.local, mine.misses = None, 0
        self.stats["dropped"] += 1

    # -------------------------------------------------------------- frames

    def frame(self, rtp: int, found: dict[str, list], now_ms: float) -> list[dict[str, Any]]:
        """One frame's boxes ({kind: [(x1, y1, x2, y2, score)]}) and its RTP
        timestamp. Returns the packet to send for it, if any."""
        when = self.clock.frame_wall(rtp, now_ms)
        if when is None:
            self.stats["untimed"] += 1
            return []
        self.stats["frames"] += 1
        seen = {t.id for t in self.tracker.update(when, found)}
        entries = []
        for mine in self.theirs.values():
            if mine.local is None or when <= mine.sent:
                continue
            local: Track | None = self.tracker.tracks.get(mine.local)
            if local is None:
                self._unbind(mine)
                continue
            if when - mine.reported > END_MS:
                continue  # Protect no longer reports it
            if local.id in seen:
                box = local.box
            elif when - local.seen <= COAST_MS:
                box = local.expected(when)
            else:
                continue
            entries.append(self._entry(mine, box))
            mine.sent = when
        if not entries:
            return []
        self.stats["sent"] += 1
        return [self._packet(when, entries)]

    # ------------------------------------------------------------- packets

    def _entry(self, mine: _Theirs, box) -> dict[str, Any]:
        return {"id": mine.id, "coord": to_coord(box), **mine.meta, "stationary": False, "local": True}

    def _packet(self, when: float, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        gap = (self.clock.wall or 0.0) - (self.clock.stream or 0.0)
        return {
            "source": "local",
            "stampMs": 0,  # these boxes are stamped with their own frame's time
            "clockWall": round(when, 1),
            "clockStream": round(when - gap, 1),
            "clockStreamRate": 1000,
            **self.event,
            "tracks": tracks,
        }


def restamp(packet: dict[str, Any], stamp_ms: float) -> dict[str, Any]:
    """One of Protect's reports, stamped with the time of the frame it
    describes, as the local packets are (so the two can be mixed)."""
    out = dict(packet)
    rate = float(packet.get("clockStreamRate") or 1000)
    if isinstance(packet.get("clockWall"), (int, float)):
        out["clockWall"] = packet["clockWall"] + stamp_ms
    if isinstance(packet.get("clockStream"), (int, float)):
        out["clockStream"] = packet["clockStream"] + stamp_ms * rate / 1000.0
    out["stampMs"] = 0
    out["source"] = "protect"
    return out
