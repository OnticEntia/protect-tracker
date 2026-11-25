"""Is the whole view moving? Told from made-up pictures (scenes.py): a steady
view with things moving in it, and a view that pans, tilts, jumps and zooms."""

import numpy as np

from scenes import scene, view, with_thing
from tracker_service.view import H, MOVING, STEADY, SUSPECT, W, ViewMotion

S = scene()
X, Y = 200.0, 120.0  # where the view rests at first


def run(picture_at, seconds: float, fps: int = 10, motion: ViewMotion | None = None) -> list[tuple[float, str]]:
    """The states, sample by sample: [(seconds, state)]."""
    motion = motion or ViewMotion()
    return [(i / fps, motion.frame(i * 1000.0 / fps, picture_at(i / fps, i))) for i in range(int(seconds * fps))]


def moving_between(states) -> tuple[float, float] | None:
    """From when to when the view was told as moving (None: never)."""
    times = [t for t, state in states if state == MOVING]
    return (times[0], times[-1]) if times else None


def spells(states) -> int:
    """How many separate times it was told as moving."""
    return sum(1 for (_, a), (_, b) in zip([(0, STEADY)] + states, states) if b == MOVING and a != MOVING)


def steady(t, i):
    return view(S, X, Y, seed=i)


def ramp(t: float, start: float, end: float) -> float:
    """Seconds of moving done by time t, for a move from `start` to `end`."""
    return min(max(t - start, 0.0), end - start)


def test_a_steady_view_is_steady():
    assert all(state == STEADY for _, state in run(steady, 10))


def test_someone_walking_through_is_not_the_view_moving():
    states = run(lambda t, i: with_thing(steady(t, i), 0.1 * t, 0.45, 0.08, 0.35), 9)
    assert all(state == STEADY for _, state in states)


def test_a_big_vehicle_filling_much_of_the_picture_is_not_the_view_moving():
    # a lorry seven tenths of the picture wide and nearly as tall as it, driving through from the left
    states = run(lambda t, i: with_thing(steady(t, i), -0.7 + 0.25 * t, 0.15, 0.7, 0.85), 8)
    assert moving_between(states) is None
    # ... nor one standing in the picture for a while and driving off (the slow comparisons must not add its way up)
    states = run(lambda t, i: with_thing(steady(t, i), 0.15 + 0.3 * ramp(t, 4, 8), 0.2, 0.6, 0.75), 12)
    assert moving_between(states) is None


def test_a_pan_is_told_soon_and_ends_soon():
    for speed in (0.1, 0.3):  # shares of the picture's width a second
        states = run(lambda t, i: view(S, X + speed * W * ramp(t, 2, 5), Y, seed=i), 9)
        began, ended = moving_between(states)
        assert 2.0 < began <= 2.5 and 5.0 <= ended <= 6.2, (speed, began, ended)
        assert spells(states) == 1  # once, not on and off


def test_a_tilt_is_told():
    states = run(lambda t, i: view(S, X, Y + 0.1 * H * ramp(t, 2, 5), seed=i), 9)
    began, ended = moving_between(states)
    assert 2.0 < began <= 2.6 and 5.0 <= ended <= 6.2 and spells(states) == 1


def test_a_pan_so_slow_that_nothing_shows_from_one_moment_to_the_next():
    for speed, by in ((0.03, 3.0), (0.01, 4.5)):
        states = run(lambda t, i: view(S, X + speed * W * ramp(t, 2, 9), Y, seed=i), 16)
        began, ended = moving_between(states)
        assert 2.0 < began <= by and 9.0 <= ended <= 14.0, (speed, began, ended)
        assert spells(states) == 1


def test_a_jump_to_another_view():
    states = run(lambda t, i: view(S, X if t < 3 else X + 220, Y if t < 3 else Y + 80, seed=i), 8)
    began, ended = moving_between(states)
    assert 3.0 <= began <= 3.3  # believed at the second look
    assert ended <= 4.5 and spells(states) == 1  # and at rest again within a second and a half
    assert states[30][1] == SUSPECT  # the first look only suspects it: nothing new is told from there on


def test_zooming_is_told_though_the_middle_stays_where_it_is():
    for rate in (0.1, 0.2):
        states = run(lambda t, i: view(S, X, Y, zoom=1 + rate * ramp(t, 2, 4), seed=i), 9)
        began, ended = moving_between(states)
        assert 2.0 < began <= 3.2 and 4.0 <= ended <= 6.5 and spells(states) == 1, (rate, began, ended)


def test_a_change_of_light_is_not_a_change_of_view():
    # the picture suddenly darker and flatter (a lamp going out, the exposure changing)
    states = run(lambda t, i: steady(t, i) * (1.0 if t < 3 else 0.55) + (0 if t < 3 else 20), 8)
    assert moving_between(states) is None
    # a noisy picture at night
    assert moving_between(run(lambda t, i: view(S, X, Y, noise=8, seed=i) * 0.4, 8)) is None


def test_nothing_to_go_by_says_nothing():
    blank = lambda t, i: np.full((H, W), 12.0, np.float32) + np.random.default_rng(i).normal(0, 1.0, (H, W))  # noqa: E731
    motion = ViewMotion()
    assert all(state == STEADY for _, state in run(blank, 6, motion=motion))
    assert motion.stats["blind"] > 0  # and that is counted


def test_what_keeps_changing_everywhere_is_not_a_view_moving_for_ever():
    """Every block changing all the time (snow in the lamps at night): told
    as moving at worst for about half a minute, then left out."""
    def flicker(t, i):
        return np.random.default_rng(i).integers(0, 255, (H, W)).astype(np.float32)

    motion = ViewMotion()
    states = run(flicker, 90, motion=motion)
    assert all(state != MOVING for t, state in states if t > 45)


def test_it_costs_next_to_nothing():
    import time

    motion = ViewMotion()
    pictures = [view(S, X + i, Y, seed=i).astype(np.uint8) for i in range(100)]
    started = time.perf_counter()
    for i, pic in enumerate(pictures):
        motion.frame(i * 100.0, pic)
    assert (time.perf_counter() - started) / len(pictures) < 0.005  # (it is about a third of a millisecond; this is slack for a busy machine)
