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
box of a parked car is steadier than a fresh guess thirty times a second. And
a local track is only bound while it is itself going somewhere, so a parked
thing Protect calls moving for one report is not taken up either. Nor is a
thing small and far off: the detector finds it only now and then, so a box
of ours would come and go where Protect's holds steady; it is taken up when
it has come nearer, and handed back to Protect when it has gone small again.

Protect's `stationary` is passed on as Protect said it. What our own tracker
sees is a field of its own, `moving`: has the thing's middle gone anywhere in
the last second.

One thing is added to what Protect reports: the face of a person being
followed, where the face detector finds one and Protect reports no face
there. It goes out as a track of its own, `<the person's id>:face`, with
`of` naming the person.

What goes out is stamped as Protect would have stamped it (the frame's time
less the stamp), so a card times these boxes exactly as it times Protect's,
and the two can be mixed.

Inside, times are a clock of this camera's own: the frames' RTP timestamps
counted on from the first frame. It runs whether or not Protect has reported
anything yet, so things are followed from the moment they are seen; Protect's
clock is tied to it (`shift`) as soon as a report has come.

Plain Python: no video, no model, no network. Fed by camera.py.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .clock import FeedClock
from .detect import iou
from .faces import FaceCarry, whose
from .local import LocalView
from .rtp import Unwrap
from .tracks import FAR_HEIGHT, STANDING_KINDS, LocalTracker, Track

KINDS = {"person": "person", "vehicle": "vehicle", "animal": "animal", "face": "face"}  # Protect's kind -> ours
BIND_IOU = 0.4  # a local track is bound to a Protect track when they overlap this much in the report's frame
KEEP_IOU = 0.25  # ... and stays bound while they overlap at least this much
MISSES = 3  # reports in a row that disagree before a binding is dropped
COAST_MS = 300  # a bound track not found in a frame keeps moving as it did for this long
END_MS = 1500  # no report of a Protect track for this long (by frame time): it has ended
BUSY_MS = 2000  # something moved this recently: every frame is looked at
META = ("objectType", "confidence", "zones", "loiterZones", "lines", "desc")
BEAT_MS = 1000  # with nothing to tell, the local stream still says so this often (so quiet is told from dead)
SLEW = 0.1  # the local stream's wall clock is brought to Protect's at this share of real time (never a jump, never backwards)
SEEN_MS, SEEN_MOVE = 1000, 0.05  # a local person counts as seen when followed this long and this far (share of the picture)


def to_box(coord) -> tuple:
    x, y, w, h = (float(v) / 1000.0 for v in coord)
    return (x, y, x + w, y + h)


def to_coord(box) -> list:
    x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in box)
    return [round(x1 * 1000, 1), round(y1 * 1000, 1), round((x2 - x1) * 1000, 1), round((y2 - y1) * 1000, 1)]


def _middle(box) -> tuple:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _at_edge(box, margin: float = 0.03) -> bool:
    """Does a box touch the picture's border (the thing is coming into view, or leaving it)?"""
    return bool(box) and (box[0] <= margin or box[1] <= margin or box[2] >= 1 - margin or box[3] >= 1 - margin)


def _median(values: list) -> float | None:
    values = sorted(values)
    return values[len(values) // 2] if values else None


class _Theirs:
    """One of Protect's tracks, as last reported."""

    def __init__(self, track_id: str, first: float, got: float) -> None:
        self.id = track_id
        self.meta: dict[str, Any] = {}
        self.kind: str | None = None
        self.stationary = False
        self.first = first  # the frame time of its first report
        self.first_got = got  # ... and when that report reached us (our clock)
        self.first_box: tuple = ()
        self.reported = first  # the frame time of its last report
        self.local: int | None = None  # the local track bound to it
        self.misses = 0
        self.sent = 0.0  # the newest frame time a local box of it went out for
        self.counted = False  # its head start has been noted
        self.was_moving = False
        self.was_followed = False


class CameraFollow:
    def __init__(self, stamp_ms: float = -125.0, epoch: str = "0") -> None:
        self.stamp = float(stamp_ms)
        self.tracker = LocalTracker()
        self.local = LocalView(epoch, self.tracker.place_of)  # the same tracks, told on our own word (local.py)
        self.local_packets: list[dict[str, Any]] = []  # what the last frame gave for those who asked for that
        self.faces = FaceCarry()  # (for the faces sent in Protect's name: the same carrying along)
        self._local_off: float | None = None  # what is added to our clock for the local stream's wall clock
        self._local_at = 0.0  # our clock at the last frame (for bringing that to Protect's, gently)
        self._local_sent = 0.0  # ... and when the local stream last said anything
        self._local_had = False  # did its last packet have tracks in it
        self.clock = FeedClock()
        self.theirs: dict[str, _Theirs] = {}
        self.event: dict[str, Any] = {}
        self.stats = {"frames": 0, "reports": 0, "bound": 0, "dropped": 0, "let_go": {"parked": 0, "left": 0, "far": 0, "lost": 0, "disagreed": 0}, "sent": 0, "faces": 0, "people_never_reported": 0,
                      "theirs_moving": 0, "theirs_followed": 0}  # Protect's tracks seen on the move, and how many of them we followed
        self.shift: float | None = None  # Protect's wall clock minus our own clock
        self.last_frame: tuple[float, dict[str, tuple]] | None = None  # the newest frame's time, and our boxes sent for it
        self.last_report: tuple[float, dict[str, tuple]] | None = None  # the newest report's frame time, and its boxes
        self.head: dict[str, deque] = {}  # kind -> how far ahead of Protect we were, per thing (the newest 200)
        self._unwrap = Unwrap()
        self._base: float | None = None
        self._rtp: int | None = None  # the newest frame's RTP timestamp
        self._born_at: dict[int, float] = {}  # local track -> when (our wall clock) it was first found
        self._bound_ever: set[int] = set()
        self._busy = 0.0  # our clock: when something last moved

    # ----------------------------------------------------------- the clocks

    def _own(self, rtp: int, now_ms: float) -> float:
        ms = self._unwrap(rtp) / 90.0
        if self._base is None:
            self._base = now_ms - ms
        return self._base + ms

    def _tie(self, now_ms: float) -> None:
        """Tie Protect's clock to ours, from the newest frame."""
        if self._rtp is None or self._base is None:
            return
        wall = self.clock.frame_wall(self._rtp, now_ms)
        if wall is not None:
            self.shift = wall - (self._base + self._unwrap._last / 90.0)  # noqa: SLF001

    # ------------------------------------------------------------- reports

    def report(self, packet: dict[str, Any], now_ms: float) -> list[dict[str, Any]]:
        """One of Protect's live reports. Returns packets to send (the boxes
        already found of a track that has just been bound)."""
        wall, stream = packet.get("clockWall"), packet.get("clockStream")
        if not isinstance(wall, (int, float)) or not isinstance(stream, (int, float)):
            return []
        self.clock.report(wall, stream, packet.get("clockStreamRate"), now_ms)
        self.stats["reports"] += 1
        self._tie(now_ms)
        if any(isinstance(t, dict) and not t.get("stationary") for t in packet.get("tracks") or []):
            self._busy = max(self._busy, self._newest())  # Protect says something moves
        if self.shift is None:
            return []  # no frame yet to tie the clocks with: the next report will do
        frame_time = wall + self.stamp - self.shift  # the frame these boxes belong to, on our clock
        self.event = {"eventId": packet.get("eventId"), "edgeType": packet.get("edgeType")}
        boxes: dict[str, tuple] = {}
        for tr in packet.get("tracks") or []:
            if not isinstance(tr, dict) or not isinstance(tr.get("coord"), list) or len(tr["coord"]) != 4:
                continue
            tid = str(tr.get("id"))
            mine = self.theirs.get(tid) or self.theirs.setdefault(tid, _Theirs(tid, frame_time, now_ms))
            mine.meta = {key: tr.get(key) for key in META}
            mine.meta["eventId"] = packet.get("eventId")  # the event this track was reported under
            mine.kind = KINDS.get(tr.get("objectType"))
            mine.stationary = bool(tr.get("stationary"))
            mine.reported = frame_time
            boxes[tid] = to_box(tr["coord"])
            if mine.stationary and mine.kind in STANDING_KINDS:
                self.tracker.stands(mine.kind, boxes[tid], frame_time)  # Protect says it stands there: remembered
            if not mine.first_box:
                mine.first_box = boxes[tid]
            if not mine.stationary and mine.kind is not None and not mine.was_moving:
                mine.was_moving = True
                self.stats["theirs_moving"] += 1
        for tid in [tid for tid, t in self.theirs.items() if frame_time - t.reported > END_MS]:
            del self.theirs[tid]
        self.last_report = (frame_time, boxes)

        # bindings that still hold, and those that don't
        for tid, box in boxes.items():
            mine = self.theirs[tid]
            if mine.local is None:
                continue
            local = self.tracker.tracks.get(mine.local)
            ours = local.at(frame_time) if local is not None else None
            if mine.stationary or local is None:
                self._unbind(mine, "parked" if mine.stationary else "lost")
            elif box[3] - box[1] < FAR_HEIGHT * 0.8:
                self._unbind(mine, "far")  # gone small into the distance: Protect's box from here on
            elif ours is not None and iou(ours, box) >= KEEP_IOU:
                mine.misses = 0
            else:
                mine.misses += 1
                # (ours stands still while Protect's goes on elsewhere: that is not the same thing, at once)
                if mine.misses >= MISSES or not local.moving():
                    self._unbind(mine, "disagreed")

        for mine in self.theirs.values():  # Protect's details go with our track, and are kept up
            local = self.tracker.tracks.get(mine.local) if mine.local is not None else None
            if local is not None and mine.id in boxes:
                self.local.attach(local, mine.id, mine.meta, mine.stationary)

        # new bindings: the best overlaps first, each local track for one of theirs
        taken = {t.local for t in self.theirs.values() if t.local is not None}
        pairs = []
        for tid, box in boxes.items():
            mine = self.theirs[tid]
            if mine.local is not None or mine.kind is None or mine.stationary:
                continue
            if box[3] - box[1] < FAR_HEIGHT and mine.kind != "face":
                continue  # small and far off: left to Protect until it is nearer
            for local in self.tracker.tracks.values():
                if local.kind != mine.kind or local.id in taken or not local.moving():
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
            self._bound_ever.add(lid)
            self.stats["bound"] += 1
            if not mine.was_followed:
                mine.was_followed = True
                self.stats["theirs_followed"] += 1
            local = self.tracker.tracks[lid]
            self.local.attach(local, mine.id, mine.meta, mine.stationary)
            self._head_start(mine, local)
            # what was found of it since that frame goes out now, in order
            start = max(frame_time, mine.sent)
            for when, box in local.history:
                if when > start:
                    out.append(self._packet(when, [self._entry(mine, box, True)]))
                    mine.sent = when
        self.stats["sent"] += len(out)
        return out

    def _unbind(self, mine: _Theirs, why: str) -> None:
        if why == "parked" and mine.local is not None:
            self.local.parked(mine.local)
        mine.local, mine.misses = None, 0
        self.stats["dropped"] += 1
        self.stats["let_go"][why] += 1

    def _head_start(self, mine: _Theirs, local: Track) -> None:
        """How far ahead of Protect we were with this thing (noted once)."""
        if mine.counted or not local.born_box or not mine.first_box:
            return
        mine.counted = True
        born_at = self._born_at.get(local.id)
        their_h = mine.first_box[3] - mine.first_box[1]
        self.head.setdefault(mine.kind or "?", deque(maxlen=200)).append({
            "frames_ms": mine.first - local.born,  # our first frame of it, before the frame of Protect's first report
            "told_ms": mine.first_got - born_at if born_at is not None else None,  # we knew, before Protect's first report came
            "size": (local.born_box[3] - local.born_box[1]) / their_h if their_h > 0 else None,  # its height then, against at Protect's first report
            "entered": _at_edge(local.born_box),  # first seen at the picture's border: it came into view (not a parked thing that set off)
        })

    # -------------------------------------------------------------- frames

    def frame(self, rtp: int, found: dict[str, list], now_ms: float) -> list[dict[str, Any]]:
        """One frame's boxes ({kind: [(x1, y1, x2, y2, score)]}) and its RTP
        timestamp. Returns the packet to send for it, if any."""
        when = self._own(rtp, now_ms)
        self._rtp = rtp
        self.stats["frames"] += 1
        seen_tracks = self.tracker.update(when, found)
        seen = {t.id for t in seen_tracks}
        for t in seen_tracks:
            if t.id not in self._born_at:
                self._born_at[t.id] = now_ms
                if t.kind == "person":
                    self._busy = max(self._busy, when)  # someone new: worth a close look at once
            # (a parked car's box trembles, and is lost and found again: only going somewhere counts)
            if t.kind in ("person", "vehicle", "animal") and t.moving():
                self._busy = max(self._busy, when)
        for t in self.tracker.ended:
            self._born_at.pop(t.id, None)
            if t.id in self._bound_ever:
                self._bound_ever.discard(t.id)
                for mine in self.theirs.values():
                    if mine.local == t.id:  # it went out of view, or was lost in the middle of it
                        # out at the border; or gone small into the distance; or lost in plain view
                        small = (t.box[3] - t.box[1]) < FAR_HEIGHT
                        self._unbind(mine, "left" if _at_edge(t.box) else "far" if small else "lost")
            elif t.kind == "person" and t.seen - t.born >= SEEN_MS and t.born_box:
                a, b = _middle(t.born_box), _middle(t.box)
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= SEEN_MOVE:
                    self.stats["people_never_reported"] += 1
        ours: dict[str, tuple] = {}
        self.last_frame = (when, ours)
        if self.shift is None:
            self._tie(now_ms)
        told = self.local.frame(when, self.tracker.tracks, seen_tracks, self.tracker.ended)
        self.local_packets = []
        # every packet is the whole of what is told just now; an empty one when the last thing has gone, and now and then
        if told or self._local_had or when - self._local_sent >= BEAT_MS:
            self.local_packets = [self._local_packet(when, told)]
            self._local_sent, self._local_had = when, bool(told)
        if self.shift is None:
            return []  # followed all the same; in Protect's name nothing can go out before Protect has reported
        entries = []
        people = []
        for mine in self.theirs.values():
            if mine.local is None or when <= mine.sent:
                continue
            local: Track | None = self.tracker.tracks.get(mine.local)
            if local is None:
                self._unbind(mine, "lost")
                continue
            if when - mine.reported > END_MS:
                continue  # Protect no longer reports it
            if local.id in seen:
                box = local.box
            elif when - local.seen <= COAST_MS:
                box = local.expected(when)
            else:
                continue
            entries.append(self._entry(mine, box, local.moving()))
            ours[mine.id] = tuple(box)
            mine.sent = when
            if mine.kind == "person":
                people.append((mine, box))
        # the faces of the people followed, where Protect reports none
        if people:
            bound = {t.local for t in self.theirs.values() if t.local is not None}
            faces = [f for f in seen_tracks if f.kind == "face" and f.id not in bound]
            theirs = whose([box for _, box in people], [f.box for f in faces])
            for i, (mine, box) in enumerate(people):
                face_box, score, held = None, 0.0, False
                if i in theirs:
                    face = faces[theirs[i]]
                    face_box, score = face.box, float(face.score)
                    self.faces.find(mine.id, box, face.box, when, score)
                if face_box is None:  # not found in this frame: carried along with the person since the last find
                    carried = self.faces.carry(mine.id, box, when)
                    if carried is None:
                        continue
                    (face_box, score), held = carried, True
                entries.append({
                    "id": f"{mine.id}:face", "coord": to_coord(face_box), "objectType": "face",
                    "confidence": int(round(score * 100)), "stationary": mine.stationary, "moving": True,
                    "zones": [], "loiterZones": [], "lines": [], "desc": "", "local": True, "held": held, "of": mine.id,
                })
                self.stats["faces"] += 1
        if not entries:
            return []
        self.stats["sent"] += 1
        return [self._packet(when, entries)]

    def reported_moving(self, within_ms: float = 400.0) -> list[tuple[str, tuple]]:
        """What Protect reports as on the move just now: [(kind, box)], from
        its newest report if that is fresh."""
        if self.last_report is None or self._newest() - self.last_report[0] > within_ms + 600:
            return []
        boxes = self.last_report[1]
        return [(t.kind, boxes[tid]) for tid, t in self.theirs.items() if tid in boxes and t.kind and not t.stationary]

    def busy(self) -> bool:
        """Is something on the move (so every frame is worth a look)?"""
        if self._base is None:
            return True
        return self._newest() - self._busy <= BUSY_MS

    def _newest(self) -> float:
        """The newest frame's time on our clock (0 before the first)."""
        return 0.0 if self._base is None else self._base + (self._unwrap._last or 0) / 90.0  # noqa: SLF001

    def head_start(self) -> dict[str, Any]:
        out = {}
        for kind, all_rows in self.head.items():
            # vehicles: only those that came into view; one we had all along (a parked car setting off) would count
            # for far too much. People are never parked: every one counts.
            rows = [r for r in all_rows if r.get("entered") or kind in ("person", "face")]
            out[kind] = {
                "n": len(rows),
                "n_all": len(all_rows),
                "frames_ms": _median([round(r["frames_ms"]) for r in rows]),
                "told_ms": _median([round(r["told_ms"]) for r in rows if r["told_ms"] is not None]),
                "size": _median([round(r["size"], 2) for r in rows if r["size"] is not None]),
            }
        return out

    # ------------------------------------------------------------- packets

    def _local_packet(self, when: float, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        """A packet of the local stream. Its stream clock is the frames' own
        from the first packet (that is what a card matches the picture with).
        Its wall clock is ours until Protect's is known, and is then brought
        to Protect's little by little: it never jumps and never runs back."""
        if self._local_off is None:
            self._local_off = self.shift if self.shift is not None else 0.0
        elif self.shift is not None and self._local_off != self.shift:
            most = SLEW * max(0.0, when - self._local_at)
            self._local_off += max(-most, min(most, self.shift - self._local_off))
        self._local_at = when
        wall = when + self._local_off - self.stamp
        stream = (self._unwrap._last or 0) / 90.0 - (self._newest() - when) - self.stamp  # noqa: SLF001
        return {
            "source": "local", "mode": "local", "snapshot": True,
            "clockWall": round(wall, 1), "clockStream": round(stream, 1), "clockStreamRate": 1000,
            "eventId": None, "tracks": tracks,
        }

    def _entry(self, mine: _Theirs, box, moving: bool) -> dict[str, Any]:
        # `stationary` is Protect's word, as it last said it; `moving` is what we see ourselves
        return {"id": mine.id, "coord": to_coord(box), **mine.meta, "stationary": mine.stationary, "moving": moving, "local": True}

    def _packet(self, when: float, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        wall = when + (self.shift or 0.0) - self.stamp  # as Protect would have stamped these boxes
        if self.shift is None:
            # Protect has said nothing yet, so its wall clock is not known: ours stands in for it. The stream
            # clock is the frames' own (what a card matches the picture with), so this is still exact.
            gap = wall - ((self._unwrap._last or 0) / 90.0 - (self._newest() - when) - self.stamp)  # noqa: SLF001
        else:
            gap = (self.clock.wall or 0.0) - (self.clock.stream or 0.0)
        return {
            "source": "local",
            "clockWall": round(wall, 1),
            "clockStream": round(wall - gap, 1),
            "clockStreamRate": 1000,
            **self.event,
            "tracks": tracks,
        }
