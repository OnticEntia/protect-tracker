"""A person's face, kept with them between the frames it is found in.

The face detector does not find a face in every frame: the head turns, the
picture blurs, the person is small. Sending a face box only for the frames
with a find leaves whoever draws it holding the last one, and a walking
person's face is out of a held box within a few frames: the box trails.

So the face is remembered relative to its person: how far from the top
middle of the person's box it sat, and how big it was, when it was last
found. For every frame the person's box moves, the face's box moves with it;
a find corrects it. A second without a find and it is given up: a face not
seen that long is no longer known to be there.

How well that does is counted as it goes: at each find, how far the carried
box was from where the face was found (in face widths), and how far a box
simply left where it was last found would have been.

Plain Python.
"""

from __future__ import annotations

from collections import deque
from typing import Any

HOLD_MS = 1000  # a face not found for this long is no longer told


def _median(values) -> float | None:
    values = sorted(values)
    return round(values[len(values) // 2], 3) if values else None


class FaceCarry:
    def __init__(self) -> None:
        self._known: dict[str, dict[str, Any]] = {}  # person -> the face as last found, relative to them
        self.found = 0  # face boxes told from a find in that very frame
        self.carried = 0  # ... and from carrying the last find along with the person
        self._carry_err: deque = deque(maxlen=2000)
        self._hold_err: deque = deque(maxlen=2000)

    @staticmethod
    def _place(state: dict[str, Any], person) -> tuple:
        """Where the remembered face is, now that its person's box is `person`."""
        width = person[2] - person[0]
        scale = max(0.7, min(1.4, width / state["width"])) if state["width"] > 0 else 1.0
        cx = (person[0] + person[2]) / 2 + state["dx"] * scale
        cy = person[1] + state["dy"] * scale
        w, h = state["w"] * scale, state["h"] * scale
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def find(self, key: str, person, face, when: float, score: float = 0.0) -> None:
        """The face of this person was found in this frame."""
        state = self._known.get(key)
        fw = face[2] - face[0]
        if state is not None and fw > 0 and 0 < when - state["when"] <= HOLD_MS and when - state["when"] > 50:
            carried, held = self._place(state, person), state["box"]
            middle = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731
            off = lambda b: (abs(middle(b)[0] - middle(face)[0]) + abs(middle(b)[1] - middle(face)[1])) / fw  # noqa: E731
            self._carry_err.append(off(carried))
            self._hold_err.append(off(held))
        self._known[key] = {
            "dx": (face[0] + face[2]) / 2 - (person[0] + person[2]) / 2, "dy": (face[1] + face[3]) / 2 - person[1],
            "w": fw, "h": face[3] - face[1], "width": person[2] - person[0], "when": when, "box": tuple(face[:4]), "score": score,
        }
        self.found += 1

    def carry(self, key: str, person, when: float) -> tuple[tuple, float] | None:
        """No face found for this person in this frame: where it is by where
        they are (and how sure the last find was), or None once it is given up."""
        state = self._known.get(key)
        if state is None:
            return None
        if when - state["when"] > HOLD_MS:
            del self._known[key]
            return None
        self.carried += 1
        return self._place(state, person), state["score"]

    def forget(self, key: str) -> None:
        self._known.pop(key, None)

    def stats(self) -> dict[str, Any]:
        return {
            "found": self.found, "carried": self.carried,
            # at a find: how far off the carried box was, and a box left where it was would have been (face widths)
            "carried_off": _median(self._carry_err), "held_off": _median(self._hold_err), "compared": len(self._carry_err),
        }
