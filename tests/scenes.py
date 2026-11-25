"""Made-up views for the tests of view motion: a wide scene with detail in it
(walls, windows, kerbs, a patch of plain sky), looked at through a window
that can be turned and zoomed, with things that move across it. Nothing here
is from a camera."""

from __future__ import annotations

import numpy as np

from tracker_service.view import H, W

SCENE_W, SCENE_H = 640, 360


def scene(seed: int = 1) -> np.ndarray:
    """A wide grey scene: blocks of different brightness with edges, lines, and a plain stretch."""
    rng = np.random.default_rng(seed)
    img = np.full((SCENE_H, SCENE_W), 110.0, np.float32)
    for _ in range(260):
        w, h = int(rng.integers(6, 70)), int(rng.integers(4, 40))
        x, y = int(rng.integers(0, SCENE_W - w)), int(rng.integers(60, SCENE_H - h))
        img[y : y + h, x : x + w] = rng.integers(30, 220)
    for _ in range(120):  # thin lines: kerbs, poles, window bars
        x, y = int(rng.integers(0, SCENE_W - 40)), int(rng.integers(60, SCENE_H - 2))
        if rng.random() < 0.5:
            img[y : y + 2, x : x + int(rng.integers(10, 40))] = rng.integers(20, 240)
        else:
            img[y : min(SCENE_H, y + int(rng.integers(8, 30))), x : x + 2] = rng.integers(20, 240)
    img[:60] = 180.0  # the sky: nothing in it
    return img


def view(scene_img: np.ndarray, x: float, y: float, zoom: float = 1.0, noise: float = 1.5, seed: int = 0) -> np.ndarray:
    """What a camera sees of the scene with its top left corner at (x, y), zoomed in by `zoom` about its middle."""
    cx, cy = x + W / 2, y + H / 2
    xs = np.clip(cx + (np.arange(W) - W / 2) / zoom, 0, SCENE_W - 1.001)
    ys = np.clip(cy + (np.arange(H) - H / 2) / zoom, 0, SCENE_H - 1.001)
    x0, y0 = xs.astype(int), ys.astype(int)
    fx, fy = (xs - x0)[None, :], (ys - y0)[:, None]  # between the scene's points, as a lens would have it
    a, b = scene_img[np.ix_(y0, x0)], scene_img[np.ix_(y0, x0 + 1)]
    c, d = scene_img[np.ix_(y0 + 1, x0)], scene_img[np.ix_(y0 + 1, x0 + 1)]
    out = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    if noise:
        out += np.random.default_rng(seed).normal(0.0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 255)


def with_thing(picture: np.ndarray, left: float, top: float, width: float, height: float, seed: int = 7) -> np.ndarray:
    """The picture with a thing in it (a patterned box; left/top/width/height as shares of the picture)."""
    out = picture.copy()
    x0, y0 = int(round(left * W)), int(round(top * H))
    w, h = max(2, int(round(width * W))), max(2, int(round(height * H)))
    pattern = np.random.default_rng(seed).integers(40, 230, (h, w)).astype(np.float32)
    pattern = np.kron(pattern[::4, ::4], np.ones((4, 4), np.float32))[:h, :w]  # panels, not specks
    xa, xb, ya, yb = max(0, x0), min(W, x0 + w), max(0, y0), min(H, y0 + h)
    if xb > xa and yb > ya:
        out[ya:yb, xa:xb] = pattern[ya - y0 : yb - y0, xa - x0 : xb - x0]
    return out
