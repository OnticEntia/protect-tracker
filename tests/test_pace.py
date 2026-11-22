"""The floor: never fewer than 20 looks a second on a camera, whatever else gives way."""

import pytest

from tracker_service.pace import MIN_FPS, Pace

FRAME = 1 / 30


def run(cameras: int, seconds: float = 20.0, look=0.007, far=0.031, faces=0.008, person=True, busy=True, other=0.0, **pace):
    """Cameras sending 30 frames a second share one detector (one look at a time). `other`: the share of the
    card taken by something else. Returns per camera: (the lowest rate over any two seconds after the start, its Pace)."""
    paces = [Pace(**pace) for _ in range(cameras)]
    steps = int(seconds / FRAME)
    arrivals = sorted((i * FRAME + c * 0.004, c) for c in range(cameras) for i in range(steps))  # the cameras a little out of step
    queues = [[] for _ in range(cameras)]
    lows = [1e9] * cameras
    states = [None] * cameras  # as it stood near the end, the cameras still sending
    t, nxt = 0.0, 0
    while nxt < len(arrivals) or any(queues):
        if not any(queues):
            t = max(t, arrivals[nxt][0])
        while nxt < len(arrivals) and arrivals[nxt][0] <= t:
            queues[arrivals[nxt][1]].append(arrivals[nxt][0])
            nxt += 1
        c = min((c for c in range(cameras) if queues[c]), key=lambda c: queues[c][0])  # whoever has waited longest
        came = queues[c].pop(0)
        plan = paces[c].plan(t - came, busy, t)
        cost = 0.001  # decoding
        if plan["look"]:
            cost += look + (far if plan["far"] else 0) + (faces if plan["faces"] and person else 0)
        t += cost / (1 - other)
        if 3 < t < seconds - 1:
            lows[c] = min(lows[c], paces[c].fps(t))
            states[c] = paces[c].state(t)
    return list(zip(lows, paces, states))


def test_the_floor_cannot_be_set_lower():
    with pytest.raises(ValueError):
        Pace(min_fps=15)
    assert Pace(min_fps=25).min_fps == 25 and Pace().min_fps == MIN_FPS == 20


@pytest.mark.parametrize("cameras", [1, 2])
def test_one_and_two_cameras_hold_every_frame(cameras):
    for low, pace, state in run(cameras):
        assert low >= 29, f"{cameras} camera(s): fell to {low} looks a second"
        assert pace.shed["frames"] == 0 and not state["below"] and state["fps"] >= 29


def test_four_cameras_give_up_the_extras_before_any_frame():
    for low, pace, state in run(4):
        assert low >= MIN_FPS, f"fell to {low}"
        assert pace.shed["far"] > 0 and pace.shed["frames"] == 0  # the far look and the faces went; no frame did
        assert not state["below"]


def test_a_busy_card_sheds_but_keeps_the_floor():
    for low, pace, state in run(2, other=0.45):  # nearly half the card is doing something else
        assert low >= MIN_FPS, f"fell to {low}"
        assert pace.shed["far"] > 0 and not state["below"]


def test_when_it_cannot_be_held_that_is_said():
    results = run(8)  # far more than one detector can look at
    assert any(state["below"] for _, _, state in results)
    for low, pace, _ in results:
        assert pace.shed["far"] > 0 and pace.shed["faces"] > 0  # everything optional went first


def test_the_quiet_time_saving_stops_at_the_floor():
    (low, pace, _), = run(1, busy=False, idle_every=6)  # asked for one look in six on a 30-a-second camera
    assert low >= MIN_FPS and pace.shed["idle"] == 0  # 30 / 20 leaves no room for it


def test_first_seconds_are_not_called_below():
    pace = Pace()
    pace.plan(0.0, True, 100.0)
    assert not pace.state(100.1)["below"]


def test_one_camera_keeps_its_faces_and_far_looks():
    """All is well on one camera: a far look takes longer than a frame lasts and a frame or two wait behind it
    for a moment. That is not falling behind, and nothing is to be left out for it."""
    (low, pace, state), = run(1)
    assert pace.shed == {"far": 0, "faces": 0, "frames": 0, "idle": 0} and low >= 29


def test_frames_arriving_in_bunches_cost_nothing():
    pace = Pace()
    t = 0.0
    for i in range(600):  # four frames at once every 133 ms, each looked at in 11 ms
        late = (i % 4) * 0.011
        t = (i // 4) * 0.1333 + late
        pace.plan(late, True, t)
    assert pace.shed["faces"] == 0 and pace.shed["far"] == 0 and pace.shed["frames"] == 0


def test_two_cameras_with_nobody_about_keep_everything():
    """Vehicles only (no faces to find): two cameras have room for every frame and every far look."""
    for low, pace, state in run(2, person=False):
        assert low >= 29 and pace.shed == {"far": 0, "faces": 0, "frames": 0, "idle": 0}
