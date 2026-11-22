"""Which frames are looked at, and how closely: never fewer than the floor.

The rule that comes first: a camera being watched has at least `min_fps` of
its frames looked at every second (20 unless set higher; it cannot be set
lower), idle or busy, however many cameras share the card. Everything else
gives way to that, in this order, as the detector falls behind:

  1. the sharper look for people far off is left out of this frame;
  2. the faces are;
  3. only then is a frame not looked at at all, and only while the rate is
     still above the floor, or once the detector is a second behind (looking
     at every frame then would only make everything later and later).

The quiet-time saving (`idle_every`: one frame in so many while nothing
moves) is likewise only taken as far as the floor allows: on a camera sending
30 frames a second with a floor of 20 that is not at all.

When the floor cannot be held (the card is busy with something else, too
many cameras, a slower machine) that is said, not hidden: `below` in what
`state` gives, which goes out with the status and on every packet.

Plain Python; the clock is handed in, so it can be tested without waiting.
"""

from __future__ import annotations

from collections import deque

MIN_FPS = 20  # the floor's own floor: it cannot be set lower
BEHIND_FAR = 1  # frames waiting beyond which the far look is left out
BEHIND_FACES = 2  # ... and the faces
BEHIND_SKIP = 4  # ... and a frame may be let go unlooked (while the rate is above the floor)
BEHIND_LATE = 30  # ... and must be (a second behind: catching up comes first)
WINDOW = 2.0  # seconds the rates are taken over


class Pace:
    def __init__(self, min_fps: float = MIN_FPS, idle_every: int = 1, far_every: int = 6) -> None:
        if min_fps < MIN_FPS:
            raise ValueError(f"the floor cannot be under {MIN_FPS} frames a second")
        self.min_fps = float(min_fps)
        self.idle_every = max(1, int(idle_every))
        self.far_every = max(1, int(far_every))
        self._count = 0
        self._came: deque = deque()  # when frames came
        self._looked: deque = deque()  # when frames were looked at
        self.shed = {"far": 0, "faces": 0, "frames": 0, "idle": 0}
        self.started: float | None = None

    def _trim(self, now: float) -> None:
        for times in (self._came, self._looked):
            while times and times[0] < now - WINDOW:
                times.popleft()

    def _span(self, now: float) -> float:
        """Over how long the rates are taken (shorter just after starting)."""
        return max(0.25, min(WINDOW, now - self.started)) if self.started is not None else WINDOW

    def stream_fps(self, now: float) -> float:
        self._trim(now)
        return len(self._came) / self._span(now)

    def fps(self, now: float) -> float:
        """Frames looked at a second, of late."""
        self._trim(now)
        return len(self._looked) / self._span(now)

    def plan(self, backlog: int, busy: bool, now: float) -> dict[str, bool]:
        """A frame has come and `backlog` more are waiting behind it. What is
        to be done with it: {"look", "far", "faces"}."""
        if self.started is None:
            self.started = now
        self._count += 1
        self._came.append(now)
        rate = self.fps(now)
        # (would letting this one go leave the last second's looks under the floor?)
        last_second = sum(1 for t in self._looked if t > now - 1.0)
        room = last_second + 1 > self.min_fps and now - self.started >= 1.0
        if backlog > BEHIND_LATE or (backlog > BEHIND_SKIP and room):
            self.shed["frames"] += 1
            return {"look": False, "far": False, "faces": False}
        if not busy and self.idle_every > 1:
            every = min(self.idle_every, max(1, int(self.stream_fps(now) // self.min_fps)))
            if every > 1 and self._count % every:
                self.shed["idle"] += 1
                return {"look": False, "far": False, "faces": False}
        far_due = self._count % self.far_every == 0
        far = far_due and backlog <= BEHIND_FAR and (rate >= self.min_fps or now - self.started < 1.0)
        faces = backlog <= BEHIND_FACES
        if far_due and not far:
            self.shed["far"] += 1
        if not faces:
            self.shed["faces"] += 1
        self._looked.append(now)
        return {"look": True, "far": far, "faces": faces}

    def state(self, now: float) -> dict:
        rate, stream = self.fps(now), self.stream_fps(now)
        settled = self.started is not None and now - self.started >= WINDOW
        return {
            "fps": round(rate, 1),
            "stream_fps": round(stream, 1),
            "floor": self.min_fps,
            # under the floor although the camera sends enough: the detector cannot keep up
            "below": bool(settled and stream >= self.min_fps and rate < self.min_fps - 0.5),
            "shed": dict(self.shed),
        }
