"""Which detector model is worth its time: the candidates compared on the
camera's own pictures.

On request the service keeps a few hundred frames of the camera it is
watching (in memory only, one every so often over a couple of minutes), runs
every model it has on each of them, and says for each how long a look takes
and how its boxes compare. Then the frames are dropped. Nothing is written
anywhere; only numbers come out.

There are no hand-drawn boxes to compare with, so the yardstick is the
biggest model at the biggest picture size (the "reference"): for each
candidate, how many of the reference's things it also finds (found), how many
of its own boxes the reference agrees with (right), and how closely the boxes
that match overlap. That measures agreement with the best of the family, not
truth: where the reference is wrong, a candidate that is right is marked
down. The split by size shows where a model gives out: small things far off.

A second yardstick comes from outside the family: what Protect was reporting
as on the move when each frame was kept. For each candidate, how many of
those it finds (`protect`). Protect's box is a fraction of a second old by
then, so a loose overlap is enough, and what Protect missed says nothing
either way.

Plain numpy besides the detector.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from .detect import Detector, iou, resize
from .video import picture

log = logging.getLogger(__name__)

SIZES = ("nano", "tiny", "s", "m", "l", "x")
WIDTHS = ((640, "fast"), (1280, "wide1280"))  # picture width -> the folder its model copies are in
KINDS = ("person", "vehicle")
KEEP = (1280, 720)  # the frames are kept at this size
REF_CONF = 0.5  # what counts as a thing: the reference at least this sure
CONF = 0.4  # a candidate's boxes at least this sure
LENIENT = 0.25  # a candidate's box is "right" when the reference has that thing at all, however unsure
PROTECT_IOU = 0.3  # a thing Protect reported counts as found when a box overlaps Protect's this much
SMALL, LARGE = 0.06, 0.15  # by height, as a share of the picture: under SMALL is far off, over LARGE is near


def band(box) -> str:
    height = box[3] - box[1]
    return "small" if height < SMALL else "large" if height >= LARGE else "medium"


def match(mine: list, theirs: list, least: float = 0.5) -> list[tuple[int, int, float]]:
    """Pairs of (mine, theirs) that overlap at least `least`, each used once, the best first."""
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(mine) for j, b in enumerate(theirs)), reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    out = []
    for overlap, i, j in pairs:
        if overlap < least:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, overlap))
    return sorted(out)


def score(candidate: list[dict], reference: list[dict], protect: list[list] | None = None) -> dict[str, Any]:
    """Each is one {kind: [(x1, y1, x2, y2, score)]} per frame, at any
    confidence. `protect`: per frame, what Protect reported as moving: [(kind, box)]."""
    out: dict[str, Any] = {}
    for kind in KINDS:
        theirs = theirs_found = 0
        for mine_all, reported in zip(candidate, protect or []):
            mine = [b for b in mine_all.get(kind, []) if b[4] >= CONF]
            for their_kind, box in reported:
                if their_kind == kind:
                    theirs += 1
                    theirs_found += any(iou(box, b) >= PROTECT_IOU for b in mine)
        things = found = boxes = right = 0
        overlaps: list[float] = []
        by_band = {name: [0, 0] for name in ("small", "medium", "large")}  # found, of
        for mine_all, ref_all in zip(candidate, reference):
            mine = [b for b in mine_all.get(kind, []) if b[4] >= CONF]
            ref = [b for b in ref_all.get(kind, []) if b[4] >= REF_CONF]
            pairs = match(mine, ref)
            hit = {j for _, j, _ in pairs}
            things += len(ref)
            found += len(pairs)
            overlaps += [o for _, _, o in pairs]
            for j, box in enumerate(ref):
                by_band[band(box)][1] += 1
                by_band[band(box)][0] += j in hit
            boxes += len(mine)
            right += len(match(mine, [b for b in ref_all.get(kind, []) if b[4] >= LENIENT], 0.4))
        out[kind] = {
            "things": things,
            "found": round(found / things, 3) if things else None,
            "right": round(right / boxes, 3) if boxes else None,
            "overlap": round(float(np.mean(overlaps)), 3) if overlaps else None,
            **{name: {"of": of, "found": round(got / of, 3) if of else None} for name, (got, of) in by_band.items()},
            "protect": {"of": theirs, "found": round(theirs_found / theirs, 3) if theirs else None},
        }
    return out


class Bench:
    def __init__(self, models: Path) -> None:
        self.models = Path(models)
        self.state = "idle"
        self.frames: list[np.ndarray] = []
        self.reported: list[list] = []  # per frame: what Protect said was on the move then
        self.want = 0
        self.every = 15
        self._count = 0
        self.result: dict[str, Any] | None = None
        self.progress = ""
        self._lock = threading.Lock()

    def candidates(self) -> list[tuple[str, int, Path]]:
        out = []
        for size in SIZES:
            for width, folder in WIDTHS:
                path = self.models / folder / f"yolox_{size}.onnx"
                if path.exists():
                    out.append((size, width, path))
        return out

    def start(self, frames: int = 240, every: int = 15) -> bool:
        with self._lock:
            if self.state in ("collecting", "running"):
                return False
            self.frames, self.reported = [], []
            self.want, self.every, self._count = max(10, min(600, int(frames))), max(1, int(every)), 0
            self.result, self.progress, self.state = None, "", "collecting"
            return True

    def offer(self, frame, reported: list | None = None) -> None:
        """A decoded frame of a watched camera (called for every one), and
        what Protect reports as on the move just now: [(kind, box)]."""
        if self.state != "collecting":
            return
        with self._lock:
            if self.state != "collecting":
                return
            self._count += 1
            if self._count % self.every:
                return
            self.frames.append(picture(frame, *KEEP))
            self.reported.append(list(reported or []))
            self.progress = f"{len(self.frames)} of {self.want} frames"
            if len(self.frames) < self.want:
                return
            self.state = "running"
        threading.Thread(target=self._run, name="bench", daemon=True).start()

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "progress": self.progress, "result": self.result}

    def _run(self) -> None:
        try:
            self.result = self.compare(self.frames, self.reported)
            self.state = "done"
        except Exception as err:  # noqa: BLE001
            log.exception("The model comparison failed")
            self.result, self.state = {"error": f"{type(err).__name__}: {err}"[:200]}, "done"
        finally:
            self.frames, self.reported = [], []  # the pictures are dropped; only the numbers stay

    def compare(self, frames: list[np.ndarray], reported: list[list] | None = None) -> dict[str, Any]:
        candidates = self.candidates()
        if not candidates:
            return {"error": "no model files"}
        small = [resize(f, KEEP[0] // 2, KEEP[1] // 2).astype(np.uint8) for f in frames]
        runs: dict[tuple[str, int], dict[str, Any]] = {}
        for size, width, path in candidates:
            self.progress = f"yolox_{size} at {width}"
            detector = Detector(path)
            pictures = frames if detector.w == KEEP[0] else small
            for p in pictures[:10]:
                detector.detect(p, LENIENT, KINDS)
            found, times = [], []
            for p in pictures:
                started = time.perf_counter()
                found.append(detector.detect(p, LENIENT, KINDS))
                times.append((time.perf_counter() - started) * 1000)
            runs[(size, width)] = {"found": found, "ms": round(float(np.median(times)), 1), "ms_95": round(float(np.percentile(times, 95)), 1), "device": detector.device}
            del detector
        ref_key = max(runs, key=lambda k: (k[1], SIZES.index(k[0])))  # the widest picture, then the biggest model
        reference = runs[ref_key]["found"]
        rows = []
        for (size, width), run in runs.items():
            rows.append({
                "model": f"yolox_{size}", "width": width, "ms": run["ms"], "ms_95": run["ms_95"], "device": run["device"],
                "reference": (size, width) == ref_key, **score(run["found"], reference, reported),
            })
        return {"frames": len(frames), "reference": f"yolox_{ref_key[0]} at {ref_key[1]}", "conf": CONF, "rows": rows}
