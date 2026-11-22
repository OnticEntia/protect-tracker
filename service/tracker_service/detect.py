"""The detectors: what is in a frame, and where.

Two models, both run through ONNX Runtime, on the graphics card when there is
one and on the CPU otherwise:

  YOLOX (Apache-2.0)  people, vehicles and animals in the whole picture
  YuNet (MIT)         a face, looked for in the square round a person's head

The model files are not part of this repository: they are read from a models
folder (see the README for where to get them).

Boxes are (x1, y1, x2, y2, score) in 0-1 units of the picture.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

# The runtime can report usage to its maker. Not from here: switched off before it is loaded, and again when it is.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

log = logging.getLogger(__name__)

# What is told apart, each a few of the model's 80 kinds (COCO ids), named as Protect names them.
KINDS = {"person": (0,), "vehicle": (1, 2, 3, 5, 7), "animal": (14, 15, 16, 17, 18, 19)}
FACE_EYES = 0.25  # a face seen from the front has its eyes at least this far apart (share of its width)
FACE_SHARE = (0.06, 0.3)  # a face is this share of its person's height


def iou(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def session(model: Path):
    """An ONNX Runtime session for a model file, and where it runs ("GPU" | "CPU")."""
    import onnxruntime as ort  # noqa: PLC0415

    ort.disable_telemetry_events()
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()  # the GPU build finds its CUDA libraries (installed as Python packages) this way
        except Exception:  # noqa: BLE001
            pass
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # idle threads sleep between frames
    wanted = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        wanted.insert(0, "CUDAExecutionProvider")
    try:
        made = ort.InferenceSession(str(model), opts, providers=wanted)
    except Exception as exc:  # noqa: BLE001 - the card is busy or its libraries are missing: the CPU still works
        log.warning("%s could not use the graphics card (%s); running on the CPU", Path(model).name, str(exc)[:120])
        made = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    return made, "GPU" if made.get_providers()[0].startswith("CUDA") else "CPU"


def fast_copy(model: Path) -> Path:
    """Where a model's fast copy lies: <models>/fast/<same name> (half
    precision, an input only as tall as a 16:9 frame needs)."""
    return Path(model).parent / "fast" / Path(model).name


def resize(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """A frame made smaller (or larger), each new point the average of the
    four round it. Good enough for a detector, and no other library needed."""
    h, w = frame.shape[:2]
    if (w, h) == (out_w, out_h):
        return frame
    xs = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5, 0, w - 1)
    ys = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5, 0, h - 1)
    xl, yl = np.floor(xs).astype(np.intp), np.floor(ys).astype(np.intp)
    xh, yh = np.minimum(xl + 1, w - 1), np.minimum(yl + 1, h - 1)
    fx, fy = (xs - xl)[None, :, None], (ys - yl)[:, None, None]
    part = frame.astype(np.float32)
    rows = part[yl] + (part[yh] - part[yl]) * fy
    return rows[:, xl] + (rows[:, xh] - rows[:, xl]) * fx


class Detector:
    """YOLOX: people, vehicles, animals.

    The model files as published take a square picture, and a 16:9 frame fills
    only the top of it. On the graphics card the model's fast copy is
    preferred where there is one (see fast_copy)."""

    def __init__(self, model: Path) -> None:
        self.path, self.fast = Path(model), False
        self.session = None
        quick = fast_copy(self.path)
        if quick.exists():
            try:
                made, device = session(quick)
                if device == "GPU":
                    self.session, self.device, self.fast = made, device, True
            except Exception as exc:  # noqa: BLE001 - the copy would not load: the original still does
                log.warning("The fast copy of %s could not be used (%s)", self.path.name, str(exc)[:120])
        if self.session is None:
            self.session, self.device = session(self.path)
        self.input = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        in_h, in_w = int(shape[2]), int(shape[3])
        self.w, self.h = in_w, int(in_w * 9 / 16) // 2 * 2  # the 16:9 frame that fits across it
        self.grid, self.stride = self.grids(in_w, in_h)
        self.canvas = np.full((1, 3, in_h, in_w), 114, dtype=np.float32)  # the frame sits at the top, grey below

    @staticmethod
    def grids(in_w: int, in_h: int):
        grids, strides = [], []
        for s in (8, 16, 32):
            rows, cols = in_h // s, in_w // s
            ys, xs = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((rows * cols, 1), s))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def detect(self, frame: np.ndarray, min_conf: float = 0.35, kinds: tuple = ("person", "vehicle")) -> dict:
        """One look at a frame (any size, h x w x 3, BGR): {kind: [box]}."""
        self.canvas[0, :, : self.h, : self.w] = resize(frame, self.w, self.h).transpose(2, 0, 1)
        out = self.session.run(None, {self.input: self.canvas})[0][0]
        return self.decode(out, self.grid, self.stride, self.w, self.h, min_conf, kinds)

    @staticmethod
    def decode(out, grid, stride, w: int, h: int, min_conf: float, kinds: tuple) -> dict:
        """The model's raw answer as boxes by kind."""
        found = {}
        for kind in kinds:
            score = out[:, 4] * out[:, [5 + c for c in KINDS[kind]]].max(1)
            keep = score >= min_conf
            if not keep.any():
                found[kind] = []
                continue
            rows, score = out[keep], score[keep]
            centre = (rows[:, :2] + grid[keep]) * stride[keep]
            size = np.exp(rows[:, 2:4]) * stride[keep]
            boxes = np.concatenate((centre - size / 2, centre + size / 2), 1)
            picked: list[int] = []
            for i in score.argsort()[::-1]:  # the best box of each overlapping bunch
                if all(iou(boxes[i], boxes[j]) < 0.5 for j in picked):
                    picked.append(i)
            found[kind] = [
                (float(max(0.0, boxes[i][0] / w)), float(max(0.0, boxes[i][1] / h)),
                 float(min(1.0, boxes[i][2] / w)), float(min(1.0, boxes[i][3] / h)), float(score[i]))
                for i in picked
            ]
        return found


class FaceDetector:
    """YuNet: faces in a 640x640 picture. It is only ever shown the square
    round a person's head (head_crop), so a face that is a few dozen pixels in
    the camera's picture arrives large enough to find."""

    SIZE = 640

    def __init__(self, model: Path) -> None:
        self.session, self.device = session(model)
        self.input = self.session.get_inputs()[0].name
        grids, strides = [], []
        for s in (8, 16, 32):
            n = self.SIZE // s
            ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def faces(self, picture: np.ndarray, min_conf: float = 0.7, frontal: bool = False) -> list[tuple]:
        """picture: 640 x 640 x 3 (BGR, 0-255). Boxes in 0-1 units of it, best
        first. `frontal`: only faces looking this way (the model also answers
        "face" for the back of a head; its eye, nose and mouth marks are then
        not laid out as a face)."""
        out = self.session.run(None, {self.input: np.ascontiguousarray(picture.transpose(2, 0, 1)[None], dtype=np.float32)})
        cls, obj, box, marks = (np.concatenate([o[0] for o in out[i : i + 3]]) for i in (0, 3, 6, 9))
        score = np.sqrt(np.clip(cls[:, 0], 0, 1) * np.clip(obj[:, 0], 0, 1))
        keep = score >= min_conf
        if not keep.any():
            return []
        box, score, grid, stride, marks = box[keep], score[keep], self.grid[keep], self.stride[keep], marks[keep]
        marks = (marks.reshape(-1, 5, 2) + grid[:, None, :]) * stride[:, None, :] / self.SIZE
        centre = (box[:, :2] + grid) * stride
        size = np.exp(box[:, 2:4]) * stride
        boxes = np.concatenate((centre - size / 2, centre + size / 2), 1) / self.SIZE
        picked: list[int] = []
        for i in score.argsort()[::-1]:
            if all(iou(boxes[i], boxes[j]) < 0.3 for j in picked):
                picked.append(i)
        if frontal:
            picked = [i for i in picked if looks_this_way(boxes[i], marks[i])]
        return [(float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3]), float(score[i])) for i in picked]


def looks_this_way(box, marks) -> bool:
    """Are a face's marks laid out as a face seen from the front (or near
    it)? marks: right eye, left eye, nose, right and left mouth corner."""
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0:
        return False
    (rx, ry), (lx, ly), (nx, ny), (_mrx, mry), (_mlx, mly) = marks
    eyes, mouth = abs(lx - rx), (mry + mly) / 2
    return bool(
        eyes >= FACE_EYES * w and min(rx, lx) - 0.1 * eyes <= nx <= max(rx, lx) + 0.1 * eyes
        and max(ry, ly) < ny + 0.05 * h and ny < mouth and mouth - (ry + ly) / 2 >= 0.2 * h
    )


def head_crop(frame: np.ndarray, box, size: int = FaceDetector.SIZE):
    """The square round a person's head, cut from the picture and scaled to
    `size`: (the square, (its left, top and side in 0-1 units of the picture's
    width, height and width)). The head of someone upright is the top eighth
    of them; the square is generous so a bowed head is still inside."""
    h, w = frame.shape[:2]
    bw, bh = (box[2] - box[0]) * w, (box[3] - box[1]) * h
    side = max(24.0, 0.45 * bh, min(bw, 0.6 * bh))
    x0, y0 = (box[0] + box[2]) / 2 * w - side / 2, box[1] * h - 0.06 * bh
    at = (np.arange(size, dtype=np.float32) + 0.5) * (side / size) - 0.5
    xs, ys = np.clip(x0 + at, 0, w - 1), np.clip(y0 + at, 0, h - 1)
    xl, yl = np.floor(xs).astype(np.intp), np.floor(ys).astype(np.intp)
    xh, yh = np.minimum(xl + 1, w - 1), np.minimum(yl + 1, h - 1)
    # only the part of the picture the square covers is touched
    part = frame[yl[0] : yh[-1] + 1, xl[0] : xh[-1] + 1].astype(np.float32)
    xl, xh, yl, yh = xl - xl[0], xh - xl[0], yl - yl[0], yh - yl[0]
    fy, fx = (ys - np.floor(ys))[:, None, None], (xs - np.floor(xs))[None, :, None]
    rows = part[yl] + (part[yh] - part[yl]) * fy
    out = rows[:, xl] + (rows[:, xh] - rows[:, xl]) * fx
    return out, (x0 / w, y0 / h, side / w)


def find_faces(detector: FaceDetector, frame: np.ndarray, people: list, min_conf: float = 0.7, most: int = 2) -> list[tuple]:
    """Faces of the biggest `most` people in the picture, in 0-1 picture
    units with their score. A face is only believed where a head can be: in
    the upper part of its person, and of a head's size."""
    h, w = frame.shape[:2]
    found = []
    for person in sorted(people, key=lambda b: -(b[3] - b[1]))[:most]:
        square, (x0, y0, side) = head_crop(frame, person, FaceDetector.SIZE // 2)
        square = square.repeat(2, 0).repeat(2, 1)  # found at half size, then doubled: a quarter of the arithmetic
        tall = person[3] - person[1]
        for fx1, fy1, fx2, fy2, conf in detector.faces(square, min_conf):
            face = (x0 + fx1 * side, y0 + fy1 * side * w / h, x0 + fx2 * side, y0 + fy2 * side * w / h)
            if FACE_SHARE[0] * tall <= face[3] - face[1] <= FACE_SHARE[1] * tall and (face[1] + face[3]) / 2 <= person[1] + 0.4 * tall:
                found.append(tuple(max(0.0, min(1.0, v)) for v in face) + (conf,))
                break  # one face a person
    return found
