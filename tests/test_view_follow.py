"""What following does while the whole view moves, and when it has come to
rest: from made-up pictures (scenes.py) through the view check (view.py) into
the follower, as the service does it."""

from scenes import scene, view
from tracker_service.follow import CameraFollow
from tracker_service.view import H, MOVING, W, ViewMotion

S = scene()
X, Y = 200.0, 120.0
FPS = 30
# things standing in the scene (in its own points): parked cars along a kerb
CARS = [(215.0, 160.0, 34.0, 16.0), (270.0, 164.0, 36.0, 17.0), (330.0, 158.0, 34.0, 16.0), (420.0, 166.0, 38.0, 18.0), (500.0, 200.0, 36.0, 17.0)]


def seen(thing, x: float, y: float, zoom: float = 1.0):
    """A thing of the scene as the detector would find it in the view at (x, y): a box in 0-1, or None if out of view."""
    cx, cy = x + W / 2, y + H / 2
    x1 = ((thing[0] - cx) * zoom + W / 2) / W
    y1 = ((thing[1] - cy) * zoom + H / 2) / H
    x2, y2 = x1 + thing[2] * zoom / W, y1 + thing[3] * zoom / H
    if x2 < 0.03 or x1 > 0.97 or y2 < 0.03 or y1 > 0.97:
        return None
    return (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2), 0.9)


def ramp(t: float, start: float, end: float) -> float:
    return min(max(t - start, 0.0), end - start)


def run(where, seconds: float, people=None, check: bool = True, follow: CameraFollow | None = None, start: float = 0.0):
    """`where(t)` -> (x, y, zoom) of the view. `people(t)` -> boxes in the
    scene's points. Returns the follower and, per frame, (t, view state,
    the tracks told on the service's own word)."""
    follow = follow or CameraFollow(epoch="v")
    motion = run.motion if follow is run.follow else ViewMotion()
    run.follow, run.motion = follow, motion
    out = []
    for i in range(int(start * FPS), int((start + seconds) * FPS)):
        t = i / FPS
        ms = t * 1000.0
        x, y, zoom = where(t)
        if check and motion.due(ms):
            motion.frame(ms, view(S, x, y, zoom, seed=i))
        state = motion.state if check else "steady"
        found = {"vehicle": [b for b in (seen(c, x, y, zoom) for c in CARS) if b]}
        if people:
            found["person"] = [b for b in (seen(p, x, y, zoom) for p in people(t)) if b]
        follow.frame(int(ms * 90) % (1 << 32), found, 1_700_000_000_000.0 + ms, state)
        told = [tr for packet in follow.local_packets for tr in packet["tracks"]]
        out.append((t, state, told if follow.local_packets else None))
    return follow, out


run.follow = run.motion = None


def kinds_told(out) -> set:
    return {tr["objectType"] for _, _, told in out for tr in told or []}


def pan(t):
    return X + 0.15 * W * ramp(t, 6, 9), Y, 1.0


def test_a_pan_past_parked_cars_tells_nothing():
    follow, out = run(pan, 16)
    assert any(state == MOVING for _, state, _ in out)
    assert kinds_told(out) == set()  # not one parked car was told as a mover
    assert follow.stats["view"]["moves"] == 1


def test_without_the_view_check_the_same_pan_would_tell_them():
    """(So the test above is worth something: these cars do pass the travel tests when the camera turns.)"""
    _, out = run(pan, 16, check=False)
    assert "vehicle" in kinds_told(out)


def test_a_jump_to_a_preset_tells_nothing():
    follow, out = run(lambda t: (X, Y, 1.0) if t < 6 else (X + 180, Y + 30, 1.0), 14)
    assert kinds_told(out) == set() and follow.stats["view"]["moves"] == 1


def test_zooming_in_on_parked_cars_tells_nothing():
    follow, out = run(lambda t: (X + 40, Y + 10, 1 + 0.15 * ramp(t, 6, 9)), 16)
    assert any(state == MOVING for _, state, _ in out) and kinds_told(out) == set()


def test_when_it_has_come_to_rest_everything_starts_afresh_and_goes_on():
    follow, out = run(pan, 5.5)
    places_before = [p["box"] for p in follow.tracker.places]
    tracks_before = set(follow.tracker.tracks)
    assert places_before  # the parked cars had been remembered where they stood
    follow, out = run(pan, 3.0, follow=follow, start=5.5)  # ... into the middle of the pan
    assert follow.view_moving and follow.tracker.places == []  # nothing is remembered while it moves
    follow, out = run(pan, 3.0, follow=follow, start=8.5)  # ... and past its end
    assert not follow.view_moving
    assert not (set(follow.tracker.tracks) & tracks_before)  # no track from before the move is left
    assert follow.stats["view"]["dropped"] >= 3
    follow, out = run(pan, 6, follow=follow, start=11.5)
    after = [p["box"] for p in follow.tracker.places]
    assert after and all(abs(a[0] - b[0]) > 0.2 for a in after for b in places_before if abs(a[1] - b[1]) < 0.02)  # remembered anew, where they are now
    assert kinds_told(out) == set()

    # and a car that then drives through the new view is told as before
    def driving(t):
        return [(680.0 - 60.0 * (t - 18.0), 175.0, 36.0, 17.0)]

    def with_car(t):
        return pan(t)

    out_all = []
    motion = run.motion
    for i in range(int(17.5 * FPS), int(23 * FPS)):
        t = i / FPS
        ms = t * 1000.0
        x, y, zoom = with_car(t)
        if motion.due(ms):
            motion.frame(ms, view(S, x, y, zoom, seed=i))
        cars = [b for b in (seen(c, x, y, zoom) for c in CARS + (driving(t) if t >= 18 else [])) if b]
        follow.frame(int(ms * 90) % (1 << 32), {"vehicle": cars}, 1_700_000_000_000.0 + ms, motion.state)
        out_all.append((t, motion.state, [tr for packet in follow.local_packets for tr in packet["tracks"]]))
    assert "vehicle" in kinds_told(out_all)


def walker(t):
    """Someone walking slowly to the right through the scene."""
    return [(250.0 + 6.0 * t, 150.0, 9.0, 30.0)]


def test_a_person_being_followed_is_held_through_the_move_under_the_same_id():
    follow, out = run(pan, 16, people=walker)
    ids = {tr["id"] for _, _, told in out for tr in told or [] if tr["objectType"] == "person"}
    assert len(ids) == 1  # one person, one track, before, during and after
    during = [told for t, state, told in out if state == MOVING and told is not None]
    assert during and all(any(tr["objectType"] == "person" for tr in told) for told in during[3:])  # told all the way through
    assert kinds_told(out) == {"person"}  # and still not one parked car
    assert follow.stats["view"]["held"] == 1 and follow.stats["view"]["kept"] == 1
    after = [told for t, _, told in out if t > 12 and told]
    assert after  # still told once the view is at rest


def test_a_person_lost_in_a_jump_is_let_go_at_once_and_not_drawn_where_they_were():
    def jump(t):
        return (X, Y, 1.0) if t < 6 else (X + 250, Y + 60, 1.0)  # the new view does not hold them

    follow, out = run(jump, 12, people=walker)
    told_at = [t for t, _, told in out for tr in told or [] if tr["objectType"] == "person"]
    assert told_at and min(told_at) < 2 and max(told_at) < 6.4  # told before; gone within moments of the jump
    assert follow.stats["view"]["held"] == 1 and follow.stats["view"]["kept"] == 0
    assert not follow.held


def test_someone_first_seen_while_the_view_moves_is_told_once_it_is_at_rest():
    def late(t):  # walks into what the camera turns to
        return [(330.0 + 6.0 * t, 150.0, 9.0, 30.0)] if t >= 6.5 else []

    follow, out = run(pan, 16, people=late)
    told_at = [t for t, state, told in out for tr in told or [] if tr["objectType"] == "person"]
    assert told_at and all(state != MOVING for t, state, told in out if any(tr["objectType"] == "person" for tr in told or []))
    assert min(told_at) < 11.5  # within a second or so of the view coming to rest


def test_protects_tracks_are_not_taken_up_while_the_view_moves():
    from test_follow import STAMP, WALL0, report, rtp_of, theirs, where

    follow = CameraFollow(STAMP)
    follow.report(report(-1000, []), WALL0 - 500)
    step, out = 100 / 3, []
    for i in range(120):
        ms = i * step
        if i % 8 == 0 and ms >= 500:
            out += follow.report(report(ms - 500, [theirs(ms - 500, tid="car", kind="vehicle")]), WALL0 + ms)
        out += follow.frame(rtp_of(ms), {"vehicle": [where(ms) + (0.9,)]}, WALL0 + ms, "moving" if ms >= 300 else "steady")
    assert out == [] and follow.stats["bound"] == 0  # nothing went out in Protect's name


def test_every_packet_says_whether_the_view_is_moving():
    from tracker_service.camera import CameraWorker
    from test_service import SquareFinder

    worker = CameraWorker(SquareFinder(), -125.0, lambda packet, mode: None)  # (not started: only what it puts on a packet)
    assert worker._rated({"tracks": []})["view"] == "steady"
    worker.view.motion.state = "suspect"  # not yet believed: not said
    assert worker._rated({"tracks": []})["view"] == "steady"
    worker.view.motion.state = "moving"
    assert worker._rated({"tracks": []})["view"] == "moving" and worker.state()["view"]["state"] == "moving"


def test_the_view_is_asked_beside_the_cameras_thread_and_holds_up_no_frame():
    import time

    from scenes import scene as make_scene
    from tracker_service.camera import ViewWatch

    wide = make_scene()
    watch = ViewWatch()
    watch.start()
    try:
        assert watch.due(0.0)
        spent = []
        for i in range(60):  # six seconds of a jump at the third, a picture every tenth
            when = i * 100.0
            assert watch.due(when) and not watch.due(when - 50) if i else True
            picture = view(wide, X if i < 30 else X + 220, Y, seed=i)
            started = time.perf_counter()
            watch.offer(when, picture)
            spent.append(time.perf_counter() - started)
            time.sleep(0.004)  # (the comparing has its own time to be done in)
            if i == 34:
                assert watch.state == "moving"
        assert watch.state == "steady" and watch.motion.stats["moves"] == 1
        assert sorted(spent)[len(spent) // 2] < 0.0005  # leaving a picture costs the camera's thread next to nothing
        assert watch.motion.stats["checks"] + watch.skipped == 60
    finally:
        watch.stop()
        watch.join(1)
    assert not watch.is_alive()
