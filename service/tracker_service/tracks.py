"""Following things from frame to frame.

The detector says what is in each frame; this says which box in this frame is
the thing that was in the last one. Each kind is followed on its own. A box
is matched to the track it overlaps most, where the track is expected to be
by now (it keeps moving as it did); what matches nothing starts a new track,
and a track not seen for a while is dropped.

What stands still is remembered. A thing that has sat in one place for a
couple of seconds (a parked car) leaves its place on a list, kept for a
minute after it was last found there, also from what Protect calls
stationary. A box found in such a place belongs to the thing parked there: a
track that is going somewhere is never given it. Without this, a car driving
past a parked one hands its track to the parked one in the moment when only
one of the two is found (one behind the other, or the detector missing a
beat), and the parked car goes on under the passing car's name.

A person is held on to harder than anything else, because a person is what
is hidden most and changes shape most: behind a post, a railing, a car; bent
over a parcel; turning round. So a person not found is kept for a few
seconds, not half of one (unless they were at the picture's border: then
they have left), and is taken up again, as the same person, by a box of a
person's size found near where they were: nearer the sooner, further off the
longer they were out of sight (reckoned in their height: a person far off
is a few pixels wide). A box the detector is not sure of, right beside
someone already found, starts no second track. And of two boxes of people
where one is a part of the other (the top half of someone behind a railing, found beside
the whole of them: much smaller, and almost wholly inside) only the bigger
is kept.

Boxes are (x1, y1, x2, y2) in 0-1 units of the picture; times are ms.

Plain Python.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .detect import iou

GONE_MS = 500  # a track not seen for this long is dropped
GONE_FAR_MS = 1500  # ... unless it is small and far off: found only now and then, and slow to go anywhere
GONE_PERSON_MS = 3000  # ... or a person in the middle of the picture: hidden for a moment, not gone
EDGE = 0.05  # a box this near the picture's border is of something coming in or going out
SURE_BESIDE = 0.5  # a box right beside someone already found is only taken for a second person when this sure
REACH_PER_S = 1.5  # a person out of sight may be found again this many of their own widths further off for each second
INSIDE = 0.8  # of two people's boxes, the smaller is dropped when this much of it lies inside the bigger ...
PART = 0.6  # ... and it is a part of a person by its size too: no more than this share of the bigger one's
# (two whole people, one passing in front of the other, are of a size: both are kept)
FAR_HEIGHT = 0.06  # a box under this share of the picture's height is far off
KEEP_MS = 3000  # how far back a track's boxes are kept (Protect's reports come about half a second late)
MATCH_IOU = 0.2  # a box belongs to a track when they overlap at least this much
NEAR = 0.6  # ... or, failing that, when it is this near (in units of the track's size) and of like size
SMOOTH_SIZE = 0.5  # how much of a new box's size is taken at once (its edges flicker more than its middle)
SMOOTH_SPEED = 0.35
SETTLED_MS = 4000  # followed this long without going anywhere: it stands there
PLACE_KEEP_MS = 60_000  # a place something stood is remembered this long after it was last found there
PLACE_IOU = 0.6  # a box is "in that place" when it overlaps the remembered one this much
STANDING_KINDS = ("vehicle",)  # what is parked. People and animals stand about, but are never "parked"


@dataclass
class Track:
    id: int
    kind: str
    box: tuple  # as shown: the middle as found, the size smoothed
    seen: float  # when it was last found
    score: float = 0.0
    speed: tuple = (0.0, 0.0)  # of its middle, per ms
    hits: int = 1
    history: deque = field(default_factory=deque)  # (when, box), oldest first
    born: float = 0.0  # when it was first found, and where
    born_box: tuple = ()

    def expected(self, when: float) -> tuple:
        """Where it should be at `when` if it keeps moving as it did."""
        dt = max(-200.0, min(400.0, when - self.seen))
        dx, dy = self.speed[0] * dt, self.speed[1] * dt
        b = self.box
        return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)

    def moving(self, within_ms: float = 1000.0, least: float = 0.01) -> bool:
        """Has its middle gone anywhere of late (a hundredth of the picture
        within the last second, as it stands)? A parked thing's box only
        trembles."""
        hist = self.history
        if len(hist) < 2:
            return False
        newest_t, newest = hist[-1]
        for t, box in hist:
            if t >= newest_t - within_ms:
                return bool(abs((box[0] + box[2]) - (newest[0] + newest[2])) / 2 + abs((box[1] + box[3]) - (newest[1] + newest[3])) / 2 >= least)
        return False

    def at(self, when: float, slack: float = 45.0) -> tuple | None:
        """Its box at a past moment (between two frames: in between); None
        when it wasn't being followed then."""
        hist = self.history
        if not hist or when < hist[0][0] - slack or when > hist[-1][0] + slack:
            return None
        before = None
        for t, box in hist:
            if t >= when:
                if before is None or t == when:
                    return box
                t0, b0 = before
                if t - t0 > 250:  # lost in between: the nearer of the two
                    return box if t - when < when - t0 else b0
                f = (when - t0) / (t - t0)
                return tuple(b0[k] + (box[k] - b0[k]) * f for k in range(4))
            before = (t, box)
        return hist[-1][1]


def _near(track_box, box, person: bool = False, unseen_ms: float = 0.0) -> float | None:
    """How far a box is from where a track should be (in the track's own
    widths), when it is near enough and of like enough size to be the same
    thing; None when it is not."""
    tw, th = track_box[2] - track_box[0], track_box[3] - track_box[1]
    bw, bh = box[2] - box[0], box[3] - box[1]
    if tw <= 0 or th <= 0 or bw <= 0 or bh <= 0:
        return None
    dx = abs((track_box[0] + track_box[2]) - (box[0] + box[2])) / 2
    dy = abs(track_box[1] - box[1]) if person else abs((track_box[1] + track_box[3]) - (box[1] + box[3])) / 2  # a person: by the head
    ratio = (bw * bh) / (tw * th)
    if person:
        # bent over, half hidden: a person's box may be a fraction of what it was, or several times. And a person is
        # narrow (far off, a hundredth of the picture across): how far they may have gone is reckoned in their
        # height, not their width, or one step sideways would make a stranger of them.
        reach = NEAR + REACH_PER_S * unseen_ms / 1000.0
        size = max(tw, th, bh)
        # (coming out from behind a post they are a sliver at first: a sixth of what they were is still them)
        ok = 0.15 < ratio < 6.5 and dx <= reach * size and dy <= max(NEAR, 0.35 * reach) * size
        return (dx + dy) / size if ok else None
    ok = 0.5 < ratio < 2.0 and dx <= NEAR * tw and dy <= NEAR * th
    return dx / tw + dy / th if ok else None


def _beside(box, other) -> bool:
    """Is a box right beside (or on) another, within about its height?"""
    size = max(other[3] - other[1], box[3] - box[1])
    return abs((box[0] + box[2]) - (other[0] + other[2])) / 2 <= 0.6 * size and abs((box[1] + box[3]) - (other[1] + other[3])) / 2 <= 0.6 * size


def _inside(small, big) -> float:
    """The share of `small` that lies inside `big`."""
    w, h = min(small[2], big[2]) - max(small[0], big[0]), min(small[3], big[3]) - max(small[1], big[1])
    area = (small[2] - small[0]) * (small[3] - small[1])
    return (w * h) / area if w > 0 and h > 0 and area > 0 else 0.0


def one_box_a_person(boxes: list) -> list:
    """People's boxes without those lying almost wholly inside a bigger one."""
    by_size = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept: list = []
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
    for box in by_size:
        if all(_inside(box, big) < INSIDE or area(box) > PART * area(big) for big in kept):
            kept.append(box)
    return kept


def _gone_after(track: Track) -> float:
    """How long a track is kept without being found."""
    box = track.box
    if track.kind == "person":
        at_edge = box[0] <= EDGE or box[1] <= EDGE or box[2] >= 1 - EDGE or box[3] >= 1 - EDGE
        return GONE_MS if at_edge else GONE_PERSON_MS
    return GONE_FAR_MS if box[3] - box[1] < FAR_HEIGHT else GONE_MS


class LocalTracker:
    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self.ended: list[Track] = []  # the tracks dropped at the last update
        self.places: list[dict] = []  # where things stand: {kind, box, seen}
        self._next = 1

    def stands(self, kind: str, box, when: float, track: int | None = None) -> None:
        """Note that something of this kind stands here (seen now). `track`:
        our own track of the thing standing there, where there is one."""
        box = tuple(box[:4])
        for place in self.places:
            if place["kind"] == kind and iou(place["box"], box) >= PLACE_IOU:
                place["box"], place["seen"] = box, max(place["seen"], when)
                if track is not None:
                    place["by"].add(track)
                return
        self.places.append({"kind": kind, "box": box, "seen": when, "by": set() if track is None else {track}})

    def place_of(self, kind: str, box) -> dict | None:
        """The remembered place a box is in, if something stands there."""
        return self._place_of(kind, box)

    def _place_of(self, kind: str, box) -> dict | None:
        best, most = None, PLACE_IOU
        for place in self.places:
            if place["kind"] == kind:
                overlap = iou(place["box"], box)
                if overlap >= most:
                    best, most = place, overlap
        return best

    def update(self, when: float, found: dict[str, list]) -> list[Track]:
        """Take in a frame's boxes ({kind: [(x1, y1, x2, y2, score)]}).
        Returns the tracks found in this frame."""
        seen_now: list[Track] = []
        self.places = [p for p in self.places if when - p["seen"] <= PLACE_KEEP_MS]
        for kind, boxes in found.items():
            if kind == "person":
                boxes = one_box_a_person(boxes)
            tracks = [t for t in self.tracks.values() if t.kind == kind]
            expected = {t.id: t.expected(when) for t in tracks}
            going = {t.id: t.moving() for t in tracks}
            stood = [self._place_of(kind, box) for box in boxes]  # the place each box is in, if something stands there
            pairs = []
            for t in tracks:
                for j, box in enumerate(boxes):
                    # a box where something stands is that thing's: only for a track that is there itself, and, if it
                    # is going somewhere, one that has stood there (the parked car driving off, not a car passing)
                    if stood[j] is not None:
                        if iou(t.box, stood[j]["box"]) < PLACE_IOU or (going[t.id] and t.id not in stood[j]["by"]):
                            continue
                    overlap = iou(expected[t.id], box)
                    if overlap >= MATCH_IOU:
                        pairs.append((overlap, t.id, j))
                    else:
                        far = _near(expected[t.id], box, kind == "person", when - t.seen)
                        if far is not None:
                            pairs.append((-1.0 - far, t.id, j))  # a last resort, after every real overlap; the nearest first
            used_t: set[int] = set()
            used_b: set[int] = set()
            for _, tid, j in sorted(pairs, reverse=True):
                if tid in used_t or j in used_b:
                    continue
                used_t.add(tid)
                used_b.add(j)
                if stood[j] is not None and not going[tid]:
                    stood[j]["by"].add(tid)  # it is the thing standing there
                self._move(self.tracks[tid], when, boxes[j])
                seen_now.append(self.tracks[tid])
            found_now = [self.tracks[tid].box for tid in used_t]
            for j, box in enumerate(boxes):
                if j not in used_b:
                    # the detector, not sure, finding the same person twice a little apart is not a second person
                    if kind == "person" and len(box) > 4 and box[4] < SURE_BESIDE and any(_beside(box, other) for other in found_now):
                        continue
                    track = Track(self._next, kind, tuple(box[:4]), when, float(box[4]) if len(box) > 4 else 0.0, born=when, born_box=tuple(box[:4]))
                    track.history.append((when, track.box))
                    self.tracks[self._next] = track
                    self._next += 1
                    seen_now.append(track)
        for t in seen_now:  # what has stood in one place for a while leaves that place on the list
            if t.kind in STANDING_KINDS and when - t.born >= SETTLED_MS and not t.moving(SETTLED_MS):
                self.stands(t.kind, t.box, when, t.id)
        self.ended = [t for t in self.tracks.values() if when - t.seen > _gone_after(t)]
        for track in self.ended:
            del self.tracks[track.id]
        return seen_now

    @staticmethod
    def _move(track: Track, when: float, box) -> None:
        dt = when - track.seen
        old = track.box
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        person = track.kind == "person"
        if dt > 0:
            vx = (cx - (old[0] + old[2]) / 2) / dt
            # (a person is found as their top half one moment and the whole of them the next: the middle of the
            # box jumps, the head does not. Where a person is going is told by the top of their box.)
            vy = (box[1] - old[1]) / dt if person else (cy - (old[1] + old[3]) / 2) / dt
            k = SMOOTH_SPEED if track.hits > 1 else 1.0
            track.speed = (track.speed[0] + (vx - track.speed[0]) * k, track.speed[1] + (vy - track.speed[1]) * k)
        w = (old[2] - old[0]) + ((box[2] - box[0]) - (old[2] - old[0])) * SMOOTH_SIZE
        h = (old[3] - old[1]) + ((box[3] - box[1]) - (old[3] - old[1])) * SMOOTH_SIZE
        # an edge that sits on the picture's border stays there: the thing is partly out of view
        x1, x2 = cx - w / 2, cx + w / 2
        y1, y2 = (box[1], box[1] + h) if person else (cy - h / 2, cy + h / 2)  # (a person's box hangs from its top)
        if box[0] <= 0.002:
            x1 = box[0]
        if box[2] >= 0.998:
            x2 = box[2]
        if box[1] <= 0.002:
            y1 = box[1]
        if box[3] >= 0.998:
            y2 = box[3]
        track.box = (x1, y1, x2, y2)
        track.seen = when
        track.hits += 1
        if len(box) > 4:
            track.score = float(box[4])
        track.history.append((when, track.box))
        while track.history and track.history[0][0] < when - KEEP_MS:
            track.history.popleft()
