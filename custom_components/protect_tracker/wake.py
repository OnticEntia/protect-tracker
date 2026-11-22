"""When the tracking service is to watch a camera, and when it may stop.

Protect is the authority on whether something is there. Its live reports for a
camera start when it tracks something and stop when it doesn't, so they are
what wakes the service; but Protect also keeps reporting things that just
stand there (every parked car, for as long as it is parked), and those alone
are no reason to watch. So: a camera is watched from the first report of
something on the move, until nothing has moved for a while.

Plain Python (no Home Assistant), so it can be tested alone.
"""

from __future__ import annotations

from typing import Any

IDLE_S = 30.0  # nothing on the move for this long: the camera is left alone again


def moving(payload: dict[str, Any]) -> bool:
    """Does this report tell of anything on the move?"""
    return any(isinstance(t, dict) and not t.get("stationary") for t in payload.get("tracks") or [])


class Waker:
    def __init__(self, idle: float = IDLE_S) -> None:
        self.idle = idle
        self.awake: dict[str, float] = {}  # camera -> when something last moved (seconds)

    def report(self, camera: str, payload: dict[str, Any], now: float) -> bool:
        """Take in a report. True when it wakes the camera."""
        if payload.get("status") or not moving(payload):
            return False
        woke = camera not in self.awake
        self.awake[camera] = now
        return woke

    def sweep(self, now: float) -> list[str]:
        """The cameras that have just gone quiet (they are asleep from here on)."""
        quiet = [cam for cam, last in self.awake.items() if now - last > self.idle]
        for cam in quiet:
            del self.awake[cam]
        return quiet

    def forget(self, camera: str) -> bool:
        return self.awake.pop(camera, None) is not None
