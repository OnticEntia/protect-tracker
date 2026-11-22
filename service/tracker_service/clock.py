"""Which moment of the camera's clock a video frame belongs to.

Protect's live reports carry two clocks for their boxes: the camera's wall
clock (clockWall, ms) and its stream clock (clockStream, in clockStreamRate
ticks a second). A frame's RTP timestamp is that same stream clock in 90 000ths
of a second, starting over every 13 hours. So the newest report pins the stream
clock to the wall clock, and any frame's timestamp then says when, on the
camera's wall clock, the frame was stamped.

A report's boxes are stamped a fixed, measured time after the frame they
describe (`stamp_ms`, per stream of the camera): the frame for a report at
clockWall T is the one whose own time is T + stamp_ms (stamp_ms is negative).

Plain Python.
"""

from __future__ import annotations

RTP_RATE = 90_000
RTP_WRAP_MS = (1 << 32) * 1000 / RTP_RATE


class FeedClock:
    """The stream clock against the wall clock, from the reports as they come."""

    def __init__(self) -> None:
        self.wall: float | None = None  # ms, camera's wall clock, of the newest report
        self.stream: float | None = None  # ms, the stream clock then
        self.at: float | None = None  # ms, our own clock when it came

    def report(self, clock_wall: float, clock_stream: float, rate: float | None, now_ms: float) -> None:
        self.wall = float(clock_wall)
        self.stream = float(clock_stream) * 1000.0 / (float(rate) if rate else 1000.0)
        self.at = float(now_ms)

    @property
    def known(self) -> bool:
        return self.wall is not None

    def frame_wall(self, rtp: int, now_ms: float, back: float = 8000.0, ahead: float = 1500.0) -> float | None:
        """The camera's wall clock (ms) for a frame with this RTP timestamp,
        seen at `now_ms` of our clock; None when it doesn't add up (some other
        clock) or nothing has been reported yet."""
        if self.wall is None or self.stream is None or self.at is None:
            return None
        about = self.stream + (now_ms - self.at)  # the stream clock now, roughly
        ms = (rtp % (1 << 32)) * 1000.0 / RTP_RATE
        stream = ms + round((about - ms) / RTP_WRAP_MS) * RTP_WRAP_MS  # the round of the clock nearest to now
        lag = about - stream
        # a frame in hand is a little older than now, never seconds newer
        if lag < -ahead or lag > back:
            return None
        return stream + (self.wall - self.stream)
