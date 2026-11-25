"""Is the whole view moving? A camera that can pan, tilt and zoom, told from
the picture itself.

When such a camera turns, everything in its picture moves at once: every
parked car "travels", every remembered place is somewhere else. Following
has to stand back while that lasts (follow.py) and start afresh when the
view has come to rest. Protect does not say when a camera is moving, so it is
read off the picture.

How. The picture is made very small and grey (160 x 90) and only its edges
are kept: where brightness steps from left to right, and where it steps from
top to bottom, each on its own (so a change of light or exposure is not a
change of view). It is cut into 60 blocks. A few times a second each block is
compared with the same block a moment ago: do its edges still sit where they
sat? A thing moving through a steady view changes the blocks it passes over
and no others, however big it is; a view that moves changes them all. So:

  the view is moving when nearly all the blocks that have edges of one kind
  in them (and are not restless anyway, see below) have changed, counting
  the blocks round the border on their own, so that zooming, which leaves
  the middle where it is, is seen too.

The two kinds of edge are judged apart because a view turning sideways leaves
a kerb or a roof line (an edge lying flat) exactly where it was: only the
upright edges show the turn, and for tilting it is the other way about.

Three comparisons are made: with the picture a third of a second ago (a turn
at any ordinary speed, a jump to a preset), a second and a half ago, and four
seconds ago (a turn so slow that nothing shows from one moment to the next).
The two slower ones go only by blocks in which the quick one has found
nothing new all that while: otherwise a lorry driving through, which changes every block
it passes within four seconds, would look like the view having moved.

  steady    nothing of the kind
  suspect   one comparison said so: not yet believed, but nothing new is
            told meanwhile (one more and it is "moving"; else steady again)
  moving    two in a row said so. It lasts until no comparison has said so
            for half a second: under a second after an ordinary turn stops,
            up to four or five after one so slow that only the slowest
            comparison saw it. (A slower comparison is not asked while its
            older picture is from before a quicker one last saw the view
            move: what it would find changed is that same move.)

Blocks that keep changing while the view is steady (leaves in the wind, a
flag, a screen, rain in the light of the camera's own lamps) are learnt to be
restless and left out. That also bounds a mistake: if every block changes
for half a minute on end (heavy snow at night), all of them are left out,
nothing can be told any more, and the view counts as steady again rather
than moving for ever. A real turn that long is told as steady too, after
that half minute; turns are not that long.

What it cannot tell: a thing that fills nearly the whole picture as it passes
looks like the view moving, for as long as it does.

Plain numpy; a comparison takes a fraction of a millisecond of work (README).
In the service it is done on a thread of its own (camera.py: ViewWatch): it
is many small steps, and beside other busy threads each step may wait its
turn, which would otherwise hold up the frame it was asked for.
"""

from __future__ import annotations

from collections import deque

import numpy as np

W, H = 160, 90  # the small picture
BW, BH = 16, 15  # a block: 10 x 6 of them
COLS, ROWS = W // BW, H // BH
EVERY_MS = 90  # a comparison this often: every third frame of a camera sending thirty a second (their times are not
# exact to the millisecond, and a hundred would as often as not make it every fourth)
LAGS_MS = (300, 1500, 4000)  # ... with the picture this long ago
TEXTURE = 3.0  # a block has edges of a kind when brightness steps that way by at least this much on the whole (of 255)
SAME = 0.5  # a block is where it was when its edges agree at least this well with a moment ago (-1..1)
EDGE_CHANGED = 0.85  # the view moves when this share of the blocks round the border have changed ...
INNER_CHANGED = 0.6  # ... and this share of those next to them (the few in the very middle are not asked: zooming leaves them be)
STIRRED = 0.15  # a block whose edges agree less than this with a moment ago has something else in it now (a thing passing)
STILL_MOVING = 0.6  # once it is moving, this much of those two shares is enough to say it still is
MIN_BLOCKS = 10  # with fewer blocks round the border to go by, nothing is said (a blank wall, darkness, fog)
SETTLE_MS = 500  # at rest for this long, by both comparisons: steady again
RESTLESS_S = 30.0  # how slowly a block is learnt to be restless, and forgiven
RESTLESS = 0.6  # a block that has changed in this share of late comparisons is left out

STEADY, SUSPECT, MOVING = "steady", "suspect", "moving"

_MIDDLE = np.zeros((ROWS, COLS), bool)
_MIDDLE[ROWS // 2 - 1 : ROWS // 2 + 1, COLS // 2 - 2 : COLS // 2 + 2] = True  # where zooming changes least
_BORDER = np.ones((ROWS, COLS), bool)
_BORDER[1:-1, 1:-1] = False
_INNER = ~_BORDER & ~_MIDDLE


def edges(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A small grey picture's edges: the steps of brightness across, and down."""
    g = grey.astype(np.float32)
    # (softened first, three points each way: an edge is then a few points wide, so that one shifted a little
    # still agrees in part with itself, and is told from something else altogether being there)
    g[:, 1:-1] = (g[:, :-2] + g[:, 1:-1] + g[:, 2:]) / 3.0
    g[1:-1, :] = (g[:-2, :] + g[1:-1, :] + g[2:, :]) / 3.0
    across, down = np.zeros_like(g), np.zeros_like(g)
    across[:, 2:-2] = g[:, 4:] - g[:, :-4]
    down[2:-2, :] = g[4:, :] - g[:-4, :]
    return across, down


def _blocks(values: np.ndarray) -> np.ndarray:
    """Sums over the blocks."""
    return values.reshape(ROWS, BH, COLS, BW).sum(axis=(1, 3))


class ViewMotion:
    def __init__(self) -> None:
        self.state = STEADY
        self.stats = {"moves": 0, "moving_ms": 0.0, "checks": 0, "blind": 0}
        self._samples: deque = deque()  # (when, detail, its energy by block)
        self._restless = np.zeros((ROWS, COLS), np.float32)
        self._last: float | None = None
        self._rest_since: float | None = None  # since when the picture has been at rest
        self._fired = [-1e12] * len(LAGS_MS)  # when each comparison last said the view moves
        self._stirred = np.full((ROWS, COLS), -1e12)  # when the quick comparison last found each block changed

    def due(self, when: float) -> bool:
        """Is it time for a comparison (so the small picture is worth making)?"""
        return self._last is None or when - self._last >= EVERY_MS or when < self._last

    def _compare(self, now, energy, when: float, lag: float):
        """Which blocks changed against the picture `lag` ago, by each kind
        of edge: [(changed, has such edges)], or None if there is none that old."""
        old = None
        for at, pics, its in self._samples:
            if at <= when - lag * 0.8:
                old = (pics, its)
            else:
                break
        if old is None:
            return None
        least = TEXTURE * TEXTURE * BW * BH
        out = []
        for kind in (0, 1):
            agree = _blocks(now[kind] * old[0][kind]) / np.sqrt(np.maximum(energy[kind] * old[1][kind], 1e-6))
            textured = (energy[kind] >= least) & (old[1][kind] >= least)
            out.append((agree < SAME, textured, textured & (agree < STIRRED)))
        return out

    def _moving(self, compared, when: float, lag: float) -> bool | None:
        """Does this comparison say the view moves? None: too little to go by."""
        if compared is None:
            return None
        ease = STILL_MOVING if self.state == MOVING else 1.0
        said = None
        for changed, textured, _ in compared:
            usable = textured & (self._restless <= RESTLESS)
            if lag > LAGS_MS[0]:  # a slower comparison: only by blocks at rest all that while
                usable &= self._stirred < when - lag - 2 * EVERY_MS
            border, inner = usable & _BORDER, usable & _INNER
            if border.sum() < MIN_BLOCKS or inner.sum() < MIN_BLOCKS // 2:
                continue
            said = bool(said) or bool(changed[border].mean() >= EDGE_CHANGED * ease and changed[inner].mean() >= INNER_CHANGED * ease)
        return said

    def frame(self, when: float, grey: np.ndarray) -> str:
        """Take in a small grey picture (H x W) of the frame at `when` (ms). Returns the state."""
        if self._last is not None and (when < self._last or when - self._last > 5000):
            self._samples.clear()  # another clock, or a long gap: nothing to compare with
            self._rest_since = None
        passed = (when - self._last) if self._last is not None and self._samples else 0.0
        self._last = when
        now = edges(grey)
        energy = (_blocks(now[0] * now[0]), _blocks(now[1] * now[1]))
        compared = [self._compare(now, energy, when, lag) for lag in LAGS_MS]
        self._samples.append((when, now, energy))
        while self._samples and self._samples[0][0] < when - LAGS_MS[-1] * 1.2:
            self._samples.popleft()
        self.stats["checks"] += 1
        changed = np.zeros((ROWS, COLS), bool)
        if compared[0] is not None:  # what keeps changing is learnt, slowly
            for was_changed, textured, _ in compared[0]:
                changed |= was_changed & textured
            k = min(1.0, passed / (RESTLESS_S * 1000.0))
            self._restless += (changed.astype(np.float32) - self._restless) * k
        verdicts = [self._moving(compared[0], when, LAGS_MS[0])]
        if compared[0] is not None:
            for _, _, stirred in compared[0]:
                self._stirred[stirred] = when
        for i in range(1, len(LAGS_MS)):
            # a slower comparison whose older picture is from before a quicker one last saw the view move says
            # nothing new: what it finds changed is that move
            stale = any(self._fired[j] >= when - LAGS_MS[i] - 2 * EVERY_MS for j in range(i))
            verdicts.append(None if stale else self._moving(compared[i], when, LAGS_MS[i]))
        for i, verdict in enumerate(verdicts):
            if verdict:
                self._fired[i] = when
        if compared[0] is not None and verdicts[0] is None:
            self.stats["blind"] += 1
        moves = any(verdicts)
        if self.state == MOVING:
            self.stats["moving_ms"] += passed
            if moves:
                self._rest_since = None
            else:
                if self._rest_since is None:
                    self._rest_since = when
                if when - self._rest_since >= SETTLE_MS:
                    self.state, self._rest_since = STEADY, None
        elif moves:
            if self.state == SUSPECT:
                self.state = MOVING
                self.stats["moves"] += 1
            else:
                self.state = SUSPECT
        else:
            self.state = STEADY
        return self.state
