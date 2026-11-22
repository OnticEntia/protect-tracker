"""One camera, on the engine's own word: what it sees, whether or not
Protect has said anything.

follow.py is the careful half (Protect decides what exists, we refine where).
This is the other one. A thing is told as soon as the engine is sure of it,
under the engine's own track id; when Protect reports the same thing its
details are added to the track already in flight (its id, kind, description,
zones, lines) and stay with it. Nothing about a track's existence or its box
waits for Protect, and nothing stops when Protect goes quiet.

With nobody to say "that is nothing", being sure is this module's own job:

  - a person the detector is sure of (SURE_AT_ONCE) is told at once: three
    frames running, a tenth of a second, wherever they are and whether or
    not they move. Someone walking into the picture is told at its edge;
  - a person it is less sure of must also be seen walking: found in most
    frames for a quarter of a second, going somewhere step by step. And one
    that has stood in one place for seconds without the detector ever being
    sure is taken for a fixture (a post, a sign, a shape in a hedge), and is
    not told;
  - anything else (a vehicle, an animal) must have been found for a moment,
    surely enough and big enough for its kind;
  - and must have travelled from where it was first found: half its own
    size at least (the box of a parked car shifts a little whenever something
    passes in front of it), and be on the move at that moment;
  - and it must not be sitting where something is known to stand (the
    tracker's memory of parked things, from Protect and from its own eyes).
    So what only stands there (a parked car, a bike on a porch) is never
    told, however often its box comes and goes; a parked car that drives off
    is told once it has left its place;
  - once told, a person is told until their track ends, also while they
    stand still. A vehicle that has come to rest (it stands in a remembered
    place and is not moving) is left out again until it moves: a car that
    has parked is no longer anything to follow.

Plain Python; fed by follow.py, which does the following itself.
"""

from __future__ import annotations

from typing import Any

from .faces import FaceCarry, whose
from .tracks import Track

# kind -> (how sure the detector must be, the least height as a share of the picture)
# (a person far off is small: the sharper look finds them from about a fortieth of the picture's height)
# (a vehicle that small is far down the road, found one moment and not the next: not told until it is nearer)
SURE = {"person": (0.45, 0.025), "vehicle": (0.5, 0.06), "animal": (0.5, 0.04)}
SURE_AT_ONCE = 0.75  # a person the detector is this sure of is told at once (three frames), moving or not
FIXTURE_MS = 4000  # a "person" in one place this long, never surely found, is taken for a fixture
PERSIST_MS = 250  # found for this long ...
PERSIST_HITS = 3  # ... and in this many frames
TRAVEL = 0.02  # ... and this far from where it was first found (share of the picture, across and down together)
TRAVEL_OWN = 0.5  # ... and at least this share of its own smaller side
SHOWN_UNSEEN_MS = 1500  # a told track not found is still told for this long (the tracker may keep a person longer,
# to know them again; but a box is not shown standing where nobody has been seen for seconds)
GONE_SMALL = 0.7  # a told vehicle or animal under this share of its kind's least height is left out again


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
    def __init__(self, epoch: str, standing=None, stands=None) -> None:
        self.epoch = epoch  # part of every track id, so ids are not used again after a restart
        self.standing = standing or (lambda kind, box: None)  # is this box where something is known to stand
        self.stands = stands or (lambda kind, box, when, track=None: None)  # note that something stands here
        self.tracks: dict[int, _Told] = {}
        self.faces = FaceCarry()  # each person's face, kept with them between the frames it is found in
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

    def why_not(self, track: Track, told: _Told) -> list[str]:
        """What still stands in the way of telling this thing (for looking into it; `sure` is the rule)."""
        need = SURE.get(track.kind)
        if need is None or not track.born_box:
            return ["not a kind that is told"]
        a, b = _middle(track.born_box), _middle(track.box)
        own = min(track.box[2] - track.box[0], track.box[3] - track.box[1])
        out = []
        if track.kind == "person":
            if told.best < need[0]:
                out.append(f"not sure enough ({told.best:.2f} < {need[0]})")
            if track.box[3] - track.box[1] < need[1]:
                out.append(f"too small ({track.box[3] - track.box[1]:.3f} < {need[1]})")
            if track.hits < PERSIST_HITS:
                out.append("not found for long enough yet")
            if self.standing("person", track.box) is not None and not track.walking():
                out.append("where a fixture is known to be")
            elif told.best < SURE_AT_ONCE:
                out.append("stands like a fixture" if track.stood(FIXTURE_MS) else "not surely a person, and not seen walking" if not track.walking() else "")
            return [w for w in out if w]
        if told.best < need[0]:
            out.append(f"not sure enough ({told.best:.2f} < {need[0]})")
        if track.box[3] - track.box[1] < need[1]:
            out.append(f"too small ({track.box[3] - track.box[1]:.3f} < {need[1]})")
        if track.hits < PERSIST_HITS or track.seen - track.born < PERSIST_MS:
            out.append("not found for long enough yet")
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < max(TRAVEL, TRAVEL_OWN * own):
            out.append(f"has not travelled ({abs(a[0] - b[0]) + abs(a[1] - b[1]):.3f} < {max(TRAVEL, TRAVEL_OWN * own):.3f})")
        if not track.moving():
            out.append("not moving just now")
        if self.standing(track.kind, track.box) is not None:
            out.append("where something stands")
        return out

    def sure(self, track: Track, told: _Told) -> bool:
        need = SURE.get(track.kind)
        if need is None or not track.born_box:
            return False
        if track.kind == "person":
            if told.best < need[0] or track.box[3] - track.box[1] < need[1] or track.hits < PERSIST_HITS:
                return False
            if self.standing("person", track.box) is not None and not track.walking():
                return False  # where a fixture is known to be (a post, a sign: found as a person time and again)
            if told.best >= SURE_AT_ONCE:
                return True  # plainly a person: told at once
            if track.stood(FIXTURE_MS):
                return False  # never surely a person, and has not stirred: a fixture
            return track.seen - track.born >= PERSIST_MS and track.walking()
        a, b = _middle(track.born_box), _middle(track.box)
        own = min(track.box[2] - track.box[0], track.box[3] - track.box[1])
        return (
            told.best >= need[0]
            and track.box[3] - track.box[1] >= need[1]
            and track.hits >= PERSIST_HITS
            and track.seen - track.born >= PERSIST_MS
            and abs(a[0] - b[0]) + abs(a[1] - b[1]) >= max(TRAVEL, TRAVEL_OWN * own)
            and track.moving()
            and self.standing(track.kind, track.box) is None
        )

    def frame(self, when: float, tracks: dict[int, Track], seen: list[Track], ended: list[Track]) -> list[dict[str, Any]]:
        """The entries to tell for this frame."""
        for t in ended:
            gone = self.tracks.pop(t.id, None)
            if gone is not None:
                self.faces.forget(gone.name)
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
            elif told.told and when - t.seen <= SHOWN_UNSEEN_MS:  # not found in this frame: where it should be
                box = t.expected(when)
            else:
                continue
            if not told.told:
                if t.kind == "person" and t.id in seen_ids and t.stood(FIXTURE_MS):
                    self.stands("person", t.box, when, t.id)  # never told, and has not stirred: remembered as a fixture
                if not self.sure(t, told):
                    continue
                told.told = True
                self.stats["told"] += 1
                self.stats["people"] += t.kind == "person"
                if told.protect is not None:
                    self.stats["confirmed"] += 1
            elif t.kind != "person" and not t.moving() and self.standing(t.kind, t.box) is not None:
                continue  # it has come to rest where things stand: nothing to follow until it moves again
            elif t.kind != "person" and box[3] - box[1] < GONE_SMALL * SURE[t.kind][1]:
                continue  # it has gone small into the distance: found one moment and not the next from here on
            entries.append(self._entry(t, told, box))
            if t.kind == "person":
                people.append((told, box))
        # the faces of the people told: found in this frame, or carried along with their person since the last find
        faces = [f for f in seen if f.kind == "face"]
        theirs = whose([box for _, box in people], [f.box for f in faces])
        for i, (told, box) in enumerate(people):
            face_box, score, held = None, 0.0, False
            if i in theirs:
                face = faces[theirs[i]]
                face_box, score = face.box, float(face.score)
                self.faces.find(told.name, box, face.box, when, score)
            if face_box is None:
                carried = self.faces.carry(told.name, box, when)
                if carried is None:
                    continue
                (face_box, score), held = carried, True
            entries.append({
                "id": f"{told.name}:face", "coord": to_coord(face_box), "objectType": "face",
                "confidence": int(round(score * 100)), "stationary": False, "moving": True,
                "zones": [], "loiterZones": [], "lines": [], "desc": "", "local": True, "held": held,
                "confirmed": told.protect is not None, "protectId": None, "eventId": None, "of": told.name,
            })
        return entries

    @staticmethod
    def _entry(track: Track, told: _Told, box) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": told.name, "coord": to_coord(box), "objectType": track.kind, "confidence": int(round(float(told.best) * 100)),
            "stationary": False, "moving": track.moving(), "zones": [], "loiterZones": [], "lines": [], "desc": "",
            "local": True, "confirmed": False, "protectId": None, "eventId": None,
        }
        if told.protect is not None:  # Protect's details, as it last gave them (every field stays there, empty at worst)
            entry.update({k: v for k, v in told.meta.items() if v is not None})
            entry.update({"protectId": told.protect, "confirmed": True, "stationary": told.stationary})
        return entry
