"""Are our boxes better placed than Protect's? Checked against the picture.

The same check the cards' live_debug does in the browser, done here for both
sets of boxes: each frame is made small and compared with the one before, and
what changed is where things move. A box that sits on a moving thing in the
right frame is full of that movement; one that lags, leads or is loose holds
less of it. So, for each thing followed, in each frame: how much of our box
is movement, and how much of Protect's box for that same frame (between two
of its reports: in between, as the cards draw it).

Only numbers are kept (two running sums); the small pictures live in memory
for a second or so, until Protect's reports round their frame are in.

Plain Python and numpy.
"""

from __future__ import annotations

from collections import deque

import numpy as np

W, H = 320, 180
CHANGED = 18  # a point changed when its brightness moved by more than this (of 255)
WAIT_MS = 1200  # a frame is judged this long after it, when Protect's reports round it are in
GAP_MS = 600  # two reports further apart than this are not joined
MOVED = 0.006  # between those two reports its middle moved at least this far (of the picture): only things on the move say anything


def density(mask: np.ndarray, box) -> float | None:
    """The share of a box (x1, y1, x2, y2 in 0-1) that changed."""
    x0, x1 = max(0, int(box[0] * W)), min(W, int(np.ceil(box[2] * W)))
    y0, y1 = max(0, int(box[1] * H)), min(H, int(np.ceil(box[3] * H)))
    if x1 <= x0 or y1 <= y0:
        return None
    return float(mask[y0:y1, x0:x1].mean())


def overlap(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class PictureCheck:
    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._frames: deque = deque()  # (when, mask, {track id: our box})
        self._theirs: dict[str, deque] = {}  # track id -> (frame time, Protect's box), oldest first
        self.n = 0
        self.ours = 0.0
        self.theirs = 0.0
        self.agree = 0.0  # how much the two boxes overlap, summed
        self.wins = 0  # frames where ours held more movement

    def report(self, frame_time: float, boxes: dict[str, tuple]) -> None:
        """Protect's boxes for a frame time."""
        for tid, box in boxes.items():
            rows = self._theirs.setdefault(tid, deque(maxlen=40))
            if not rows or frame_time > rows[-1][0]:
                rows.append((frame_time, box))

    def frame(self, when: float, gray: np.ndarray, ours: dict[str, tuple]) -> None:
        """A frame (H x W, brightness) and our boxes in it."""
        prev, self._prev = self._prev, gray
        if prev is None or prev.shape != gray.shape:
            return
        mask = np.abs(gray.astype(np.int16) - prev) > CHANGED
        self._frames.append((when, mask, ours))
        while self._frames and self._frames[0][0] < when - WAIT_MS:
            self._judge(*self._frames.popleft())
        for tid in [tid for tid, rows in self._theirs.items() if rows[-1][0] < when - 10_000]:
            del self._theirs[tid]

    def _judge(self, when: float, mask: np.ndarray, ours: dict[str, tuple]) -> None:
        for tid, box in ours.items():
            rows = self._theirs.get(tid)
            if not rows:
                continue
            before = after = None
            for row in rows:
                if row[0] <= when:
                    before = row
                elif after is None:
                    after = row
            if before is None or after is None or after[0] - before[0] > GAP_MS:
                continue
            a, b = before[1], after[1]
            if abs((a[0] + a[2]) - (b[0] + b[2])) / 2 + abs((a[1] + a[3]) - (b[1] + b[3])) / 2 < MOVED:
                continue
            f = (when - before[0]) / (after[0] - before[0])
            their_box = tuple(a[k] + (b[k] - a[k]) * f for k in range(4))
            mine, theirs = density(mask, box), density(mask, their_box)
            if mine is None or theirs is None:
                continue
            self.n += 1
            self.ours += mine
            self.theirs += theirs
            self.agree += overlap(box, their_box)
            self.wins += mine > theirs

    def result(self) -> dict:
        if not self.n:
            return {"n": 0}
        return {
            "n": self.n,
            "ours": round(self.ours / self.n, 4),  # share of our boxes that was movement
            "protect": round(self.theirs / self.n, 4),  # ... of Protect's, for the same frames
            "ours_better": round(self.wins / self.n, 3),  # share of frames where ours held more
            "overlap": round(self.agree / self.n, 3),
        }
