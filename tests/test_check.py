"""The picture check tells a box that sits on a moving thing from one that lags."""

import numpy as np

from tracker_service.check import H, W, PictureCheck


def scene(ms: float) -> tuple[np.ndarray, tuple]:
    """A textured block moving right at an even pace, and where it is."""
    x = 0.1 + 0.00015 * ms
    gray = np.full((H, W), 20, np.uint8)
    x0, x1, y0, y1 = max(0, int(x * W)), max(0, int((x + 0.15) * W)), int(0.3 * H), int(0.7 * H)
    rng = np.random.default_rng(int(abs(ms)) + 7)
    gray[y0:y1, x0:x1] = rng.integers(80, 255, (y1 - y0, x1 - x0))
    return gray, (x, 0.3, x + 0.15, 0.7)


def play(lag_ms: float, their_lag_ms: float = 0.0) -> dict:
    check = PictureCheck()
    for i in range(120):
        ms = i * 100 / 3
        gray, _ = scene(ms)
        if i % 8 == 0:  # Protect: four a second
            check.report(ms, {"t": scene(ms - their_lag_ms)[1]})
        check.frame(ms, gray, {"t": scene(ms - lag_ms)[1], "unknown": (0, 0, 0.1, 0.1)})
    return check.result()


def test_boxes_on_the_thing_hold_its_movement():
    got = play(0)
    assert got["n"] > 40 and got["ours"] > 0.7 and abs(got["ours"] - got["protect"]) < 0.05 and got["overlap"] > 0.95


def test_a_lagging_box_holds_less():
    late = play(500)
    assert late["ours"] < late["protect"] - 0.2 and late["ours_better"] < 0.1
    theirs_late = play(0, their_lag_ms=500)
    assert theirs_late["ours"] > theirs_late["protect"] + 0.2 and theirs_late["ours_better"] > 0.9


def test_still_things_say_nothing():
    check = PictureCheck()
    gray, box = scene(0)
    for i in range(120):
        ms = i * 100 / 3
        if i % 8 == 0:
            check.report(ms, {"t": box})
        check.frame(ms, gray, {"t": box})
    assert check.result() == {"n": 0}
