"""One camera, on the engine's own word: what it sees, whether or not
Protect has said anything.

follow.py is the careful half (Protect decides what exists, we refine where).
This is the other one. A thing is told as soon as the engine is sure of it,
under the engine's own track id; when Protect reports the same thing its
details are added to the track already in flight (its id, kind, description,
zones, lines) and stay with it. Nothing about a track's existence or its box
waits for Protect, and nothing stops when Protect goes quiet.

With nobody to say "that is nothing", being sure is this module's own job:

  - a thing must have been found for a moment (a quarter of a second, a few
    frames running), surely enough and big enough for its kind;
  - and it must have travelled from where it was first found. What only
    stands there (a parked car, a bike on a porch) is never told, however
    often its box comes and goes; a parked car that drives off is told from
    the moment it has moved;
  - once told, a track is told until it ends, also while it stands still.

Plain Python; fed by follow.py, which does the following itself.
"""

from __future__ import annotations

from typing import Any

from .tracks import Track

# kind -> (how sure the detector must be, the least height as a share of the picture)
# (a person far off is small: the sharper look finds them from about a fortieth of the picture's height)
SURE = {"person": (0.45, 0.025), "vehicle": (0.5, 0.04), "animal": (0.5, 0.04)}
PERSIST_MS = 250  # found for this long ...
PERSIST_HITS = 3  # ... and in this many frames
TRAVEL = 0.02  # ... and this far from where it was first found (share of the picture, across and down together)
COAST_MS = 300  # a told track not found in a frame keeps moving as it did for this long


def _middle(box) -> tuple:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def to_coord(box) -> list:
    x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in box)
    return [round(x1 * 1000, 1), round(y1 * 1000, 1), round((x2 - x1) * 1000, 1), round((y2 - y1) * 1000, 1)]


class _Told:
    """What goes with one of our tracks."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.told = False
        self.best = 0.0  # the surest the detector has been of it
        self.protect: str | None = None  # Protect's track id, once it has reported this thing
        self.meta: dict[str, Any] = {}
        self.stationary = False


class LocalView:
    def __init__(self, epoch: str) -> None:
        self.epoch = epoch  # part of every track id, so ids are not used again after a restart
        self.tracks: dict[int, _Told] = {}
        self.stats = {"told": 0, "confirmed": 0, "people": 0}

    def _of(self, track: Track) -> _Told:
        told = self.tracks.get(track.id)
        if told is None:
            told = self.tracks[track.id] = _Told(f"L{self.epoch}-{track.id}")
        return told

    def attach(self, track: Track, protect_id: str, meta: dict[str, Any], stationary: bool) -> None:
        """Protect has reported this thing: its details go with our track from here on."""
        for other in self.tracks.values():
            if other.protect == protect_id and other is not self.tracks.get(track.id):
                other.protect = None  # Protect's track is this one now
        told = self._of(track)
        if told.protect is None and told.told:
            self.stats["confirmed"] += 1
        told.protect, told.meta, told.stationary = protect_id, dict(meta), stationary

    def parked(self, track_id: int) -> None:
        """Protect now calls the thing stationary."""
        told = self.tracks.get(track_id)
        if told is not None and told.protect is not None:
            told.stationary = True

    def sure(self, track: Track, told: _Told) -> bool:
        need = SURE.get(track.kind)
        if need is None or not track.born_box:
            return False
        a, b = _middle(track.born_box), _middle(track.box)
        return (
            told.best >= need[0]
            and track.box[3] - track.box[1] >= need[1]
            and track.hits >= PERSIST_HITS
            and track.seen - track.born >= PERSIST_MS
            and abs(a[0] - b[0]) + abs(a[1] - b[1]) >= TRAVEL
        )

    def frame(self, when: float, tracks: dict[int, Track], seen: list[Track], ended: list[Track]) -> list[dict[str, Any]]:
        """The entries to tell for this frame."""
        for t in ended:
            self.tracks.pop(t.id, None)
        seen_ids = {t.id for t in seen}
        entries: list[dict[str, Any]] = []
        people: list[tuple[_Told, tuple]] = []
        for t in tracks.values():
            if t.kind == "face":
                continue
            told = self._of(t)
            if t.id in seen_ids:
                told.best = max(told.best, float(t.score))
                box = t.box
            elif told.told and when - t.seen <= COAST_MS:
                box = t.expected(when)
            else:
                continue
            if not told.told:
                if not self.sure(t, told):
                    continue
                told.told = True
                self.stats["told"] += 1
                self.stats["people"] += t.kind == "person"
                if told.protect is not None:
                    self.stats["confirmed"] += 1
            entries.append(self._entry(t, told, box))
            if t.kind == "person":
                people.append((told, box))
        if people:  # the faces of the people told
            for face in seen:
                if face.kind != "face":
                    continue
                fx, fy = _middle(face.box)
                for told, box in people:
                    if box[0] <= fx <= box[2] and box[1] <= fy <= box[1] + 0.45 * (box[3] - box[1]):
                        entries.append({
                            "id": f"{told.name}:face", "coord": to_coord(face.box), "objectType": "face",
                            "confidence": int(round(float(face.score) * 100)), "stationary": False, "moving": face.moving(),
                            "zones": [], "loiterZones": [], "lines": [], "desc": "", "local": True,
                            "confirmed": told.protect is not None, "of": told.name,
                        })
                        break
        return entries

    @staticmethod
    def _entry(track: Track, told: _Told, box) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": told.name, "coord": to_coord(box), "objectType": track.kind, "confidence": int(round(float(told.best) * 100)),
            "stationary": False, "moving": track.moving(), "zones": [], "loiterZones": [], "lines": [], "desc": "",
            "local": True, "confirmed": False,
        }
        if told.protect is not None:  # Protect's details, as it last gave them
            entry.update({k: v for k, v in told.meta.items() if v is not None})
            entry.update({"protectId": told.protect, "confirmed": True, "stationary": told.stationary})
        return entry
