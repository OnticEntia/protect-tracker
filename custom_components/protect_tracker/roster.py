"""Which cameras there are, and what has become of each.

Cameras come and go: one is adopted, one is unplugged for a day, one is taken
out of Protect for good. Every few seconds the integration says here what
Protect has just then, and is told what changed since:

  ready      Protect has it, it is connected, and one of its streams can be had
  offline    Protect has it, but it is not connected (or Protect itself cannot
             be asked just now, and that has lasted a while)
  no_stream  Protect has it, but none of its camera entities is enabled in
             Home Assistant, so there is no stream to watch
  (gone)     Protect no longer has it: it is forgotten altogether

A camera that is missing from what Protect has is not taken for removed at
once (the list is fetched anew now and then, and the UniFi Protect
integration may be reloading): only once it has been missing a few times in a
row. While Protect cannot be asked at all nothing is concluded, except that
after a while every camera counts as offline, so that nothing goes on trying
streams nobody can vouch for.

Plain Python (no Home Assistant), so it can be tested alone.
"""

from __future__ import annotations

READY, OFFLINE, NO_STREAM = "ready", "offline", "no_stream"
GONE_CHECKS = 5  # missing this many checks in a row: removed
UNKNOWN_CHECKS = 30  # Protect not to be asked for this many checks in a row: every camera counts as offline

Change = tuple[str, str | None, str | None]  # camera, what it was, what it is (None: not there)


class Roster:
    def __init__(self, gone_checks: int = GONE_CHECKS, unknown_checks: int = UNKNOWN_CHECKS) -> None:
        self.gone_checks, self.unknown_checks = gone_checks, unknown_checks
        self.cameras: dict[str, str] = {}  # camera -> ready | offline | no_stream
        self._missing: dict[str, int] = {}
        self._unknown = 0

    def ready(self) -> list[str]:
        return [cam for cam, state in self.cameras.items() if state == READY]

    def update(self, now: dict[str, str] | None) -> list[Change]:
        """Take in what Protect has now ({camera: state}; None when Protect
        cannot be asked). Returns what changed."""
        changes: list[Change] = []
        if now is None:
            self._unknown += 1
            if self._unknown >= self.unknown_checks:
                for cam, was in self.cameras.items():
                    if was != OFFLINE:
                        self.cameras[cam] = OFFLINE
                        changes.append((cam, was, OFFLINE))
            return changes
        self._unknown = 0
        for cam, state in now.items():
            self._missing.pop(cam, None)
            was = self.cameras.get(cam)
            if was != state:
                self.cameras[cam] = state
                changes.append((cam, was, state))
        for cam in [cam for cam in self.cameras if cam not in now]:
            self._missing[cam] = self._missing.get(cam, 0) + 1
            if self._missing[cam] >= self.gone_checks:
                del self._missing[cam]
                changes.append((cam, self.cameras.pop(cam), None))
        return changes
