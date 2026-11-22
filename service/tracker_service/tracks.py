"""Following things from frame to frame.

The detector says what is in each frame; this says which box in this frame is
the thing that was in the last one. Each kind is followed on its own. A box
is matched to the track it overlaps most, where the track is expected to be
by now (it keeps moving as it did); what matches nothing starts a new track,
and a track not seen for a while is dropped.

Boxes are (x1, y1, x2, y2) in 0-1 units of the picture; times are ms.

Plain Python.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .detect import iou

GONE_MS = 500  # a track not seen for this long is dropped
KEEP_MS = 3000  # how far back a track's boxes are kept (Protect's reports come about half a second late)
MATCH_IOU = 0.2  # a box belongs to a track when they overlap at least this much
NEAR = 0.6  # ... or, failing that, when it is this near (in units of the track's size) and of like size
SMOOTH_SIZE = 0.5  # how much of a new box's size is taken at once (its edges flicker more than its middle)
SMOOTH_SPEED = 0.35


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
                return abs((box[0] + box[2]) - (newest[0] + newest[2])) / 2 + abs((box[1] + box[3]) - (newest[1] + newest[3])) / 2 >= least
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


def _near(track_box, box) -> bool:
    tw, th = track_box[2] - track_box[0], track_box[3] - track_box[1]
    bw, bh = box[2] - box[0], box[3] - box[1]
    if tw <= 0 or th <= 0 or bw <= 0 or bh <= 0:
        return False
    dx = abs((track_box[0] + track_box[2]) - (box[0] + box[2])) / 2
    dy = abs((track_box[1] + track_box[3]) - (box[1] + box[3])) / 2
    like = 0.5 < (bw * bh) / (tw * th) < 2.0
    return like and dx <= NEAR * tw and dy <= NEAR * th


class LocalTracker:
    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self.ended: list[Track] = []  # the tracks dropped at the last update
        self._next = 1

    def update(self, when: float, found: dict[str, list]) -> list[Track]:
        """Take in a frame's boxes ({kind: [(x1, y1, x2, y2, score)]}).
        Returns the tracks found in this frame."""
        seen_now: list[Track] = []
        for kind, boxes in found.items():
            tracks = [t for t in self.tracks.values() if t.kind == kind]
            expected = {t.id: t.expected(when) for t in tracks}
            pairs = []
            for t in tracks:
                for j, box in enumerate(boxes):
                    overlap = iou(expected[t.id], box)
                    if overlap >= MATCH_IOU:
                        pairs.append((overlap, t.id, j))
                    elif _near(expected[t.id], box):
                        pairs.append((overlap - 1.0, t.id, j))  # a last resort, after every real overlap
            used_t: set[int] = set()
            used_b: set[int] = set()
            for _, tid, j in sorted(pairs, reverse=True):
                if tid in used_t or j in used_b:
                    continue
                used_t.add(tid)
                used_b.add(j)
                self._move(self.tracks[tid], when, boxes[j])
                seen_now.append(self.tracks[tid])
            for j, box in enumerate(boxes):
                if j not in used_b:
                    track = Track(self._next, kind, tuple(box[:4]), when, float(box[4]) if len(box) > 4 else 0.0, born=when, born_box=tuple(box[:4]))
                    track.history.append((when, track.box))
                    self.tracks[self._next] = track
                    self._next += 1
                    seen_now.append(track)
        self.ended = [t for t in self.tracks.values() if when - t.seen > GONE_MS]
        for track in self.ended:
            del self.tracks[track.id]
        return seen_now

    @staticmethod
    def _move(track: Track, when: float, box) -> None:
        dt = when - track.seen
        old = track.box
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if dt > 0:
            vx = (cx - (old[0] + old[2]) / 2) / dt
            vy = (cy - (old[1] + old[3]) / 2) / dt
            k = SMOOTH_SPEED if track.hits > 1 else 1.0
            track.speed = (track.speed[0] + (vx - track.speed[0]) * k, track.speed[1] + (vy - track.speed[1]) * k)
        w = (old[2] - old[0]) + ((box[2] - box[0]) - (old[2] - old[0])) * SMOOTH_SIZE
        h = (old[3] - old[1]) + ((box[3] - box[1]) - (old[3] - old[1])) * SMOOTH_SIZE
        # an edge that sits on the picture's border stays there: the thing is partly out of view
        x1, x2 = cx - w / 2, cx + w / 2
        y1, y2 = cy - h / 2, cy + h / 2
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
