"""Protect's tracks followed in our frames: binding, sending, letting go."""

from tracker_service.follow import CameraFollow, to_box
from tracker_service.tracks import LocalTracker

WALL0 = 1_700_000_000_000.0
STREAM0 = 9_000_000.0  # the stream clock (ms) when the wall clock is WALL0
STAMP = -125.0
LATE = 500.0  # a report reaches us this long after the frame it describes


def rtp_of(ms: float) -> int:
    return int((STREAM0 + ms) * 90) % (1 << 32)


def where(ms: float, speed: float = 0.0002) -> tuple:
    """A walker's box at a moment: moving right at an even pace."""
    x = 0.1 + speed * ms
    return (x, 0.3, x + 0.1, 0.7)


def report(ms: float, tracks: list) -> dict:
    """What Protect says of the frame at `ms` (stamped a little after it)."""
    return {"clockWall": WALL0 + ms - STAMP, "clockStream": STREAM0 + ms - STAMP, "clockStreamRate": 1000, "eventId": "ev1", "tracks": tracks}


def theirs(ms: float, tid="p1", kind="person", stationary=False, box=None, **more) -> dict:
    x1, y1, x2, y2 = box or where(ms)
    return {"id": tid, "objectType": kind, "confidence": 80, "stationary": stationary, "zones": [3], "desc": "",
            "coord": [x1 * 1000, y1 * 1000, (x2 - x1) * 1000, (y2 - y1) * 1000], **more}


def run(follow: CameraFollow, until: float, ours=where, protect=None, start: float = 0.0, frames=True):
    """Play time forward: a frame every 33 ms, a report every 250 ms (arriving LATE). Returns what went out."""
    out = []
    step = 100 / 3
    n0, n1 = int(start / step), int(until / step)
    for i in range(n0, n1):
        ms = i * step
        due = ms - LATE
        if protect and due >= 0 and due // 250 != (due - step) // 250:
            at = (due // 250) * 250
            tracks = protect(at)
            if tracks is not None:
                out += follow.report(report(at, tracks), WALL0 + ms)
        if frames:
            boxes = ours(ms)
            found = boxes if isinstance(boxes, dict) else {"person": [boxes + (0.9,)]} if boxes else {}
            out += follow.frame(rtp_of(ms), found, WALL0 + ms)
    return out


def primed() -> CameraFollow:
    follow = CameraFollow(STAMP)
    follow.report(report(-1000, []), WALL0 - 500)  # the clocks are known before the first frame
    return follow


def test_nothing_goes_out_for_what_protect_has_not_reported():
    follow = primed()
    assert run(follow, 2000) == []
    assert len(follow.tracker.tracks) == 1  # it is followed all the same, ready to be bound


def test_bound_to_protects_track_and_sent_every_frame():
    follow = primed()
    out = run(follow, 3000, protect=lambda at: [theirs(at)])
    assert follow.stats["bound"] == 1
    times = [p["clockWall"] + STAMP - WALL0 for p in out]
    assert times == sorted(times) and len(set(times)) == len(times)  # in order, each frame once
    # from the frame of the first report on, every frame has a box: about 30 a second
    assert min(times) < 50 and max(times) > 2900 and len(times) >= 85
    for p in out:
        (tr,) = p["tracks"]
        assert tr["id"] == "p1" and tr["objectType"] == "person" and tr["zones"] == [3] and tr["local"] is True
        assert "stampMs" not in p and p["source"] == "local" and p["eventId"] == "ev1"
        assert abs(p["clockStream"] - (p["clockWall"] - WALL0 + STREAM0)) < 0.2  # stamped as Protect stamps: the same two clocks
        # the box is where the walker was in that very frame
        ms = p["clockWall"] + STAMP - WALL0
        assert abs(to_box(tr["coord"])[0] - where(ms)[0]) < 0.004


def test_the_first_boxes_are_sent_when_the_binding_is_made():
    follow = primed()
    out = run(follow, 560, protect=lambda at: [theirs(at)])  # the first report (of the frame at 0) comes at 500
    assert len(out) >= 14 and out[0]["clockWall"] + STAMP - WALL0 < 50  # the half second already seen, at once


def test_protects_parked_things_are_left_to_protect():
    follow = primed()
    still = lambda ms: (0.5, 0.5, 0.7, 0.7)  # noqa: E731
    out = run(follow, 2000, ours=still, protect=lambda at: [theirs(at, "v1", "vehicle", stationary=True, box=still(at))])
    assert out == [] and follow.stats["bound"] == 0


def test_let_go_when_they_disagree():
    follow = primed()
    # ours walks right; Protect's box is of something else entirely from 1500 on
    protect = lambda at: [theirs(at)] if at < 1500 else [theirs(at, box=(0.7, 0.1, 0.8, 0.2))]  # noqa: E731
    out = run(follow, 4000, protect=protect)
    assert follow.stats["bound"] == 1 and follow.stats["dropped"] == 1
    last = max(p["clockWall"] + STAMP - WALL0 for p in out)
    assert 2000 < last < 3000  # three reports that disagree (each half a second late), then nothing more


def test_stops_when_the_local_track_is_lost_and_binds_again():
    follow = primed()
    ours = lambda ms: where(ms) if ms < 1000 or ms > 2000 else None  # noqa: E731 - hidden for a second
    out = run(follow, 4000, ours=ours, protect=lambda at: [theirs(at)])
    times = [p["clockWall"] + STAMP - WALL0 for p in out]
    assert not [t for t in times if 1000 + 320 < t < 2000]  # it coasts a moment, then nothing
    assert [t for t in times if 1000 <= t <= 1000 + 320]
    assert max(times) > 3900 and follow.stats["bound"] == 2


def test_leaving_the_picture_is_told_from_being_lost():
    follow = primed()
    out_of_view = lambda ms: (min(0.9, 0.5 + 0.0005 * ms), 0.3, min(1.0, 0.6 + 0.0005 * ms), 0.7) if ms < 1200 else None  # noqa: E731
    run(follow, 2500, ours=out_of_view, protect=lambda at: [theirs(at, box=out_of_view(at))] if at < 1200 else None)
    assert follow.stats["let_go"]["left"] == 1 and follow.stats["let_go"]["lost"] == 0
    follow = primed()
    run(follow, 2500, ours=lambda ms: where(ms) if ms < 1200 else None, protect=lambda at: [theirs(at)])
    assert follow.stats["let_go"]["lost"] == 1 and follow.stats["let_go"]["left"] == 0


def test_stops_when_protect_stops_reporting():
    follow = primed()
    out = run(follow, 5000, protect=lambda at: [theirs(at)] if at <= 1500 else None)
    last = max(p["clockWall"] + STAMP - WALL0 for p in out)
    assert 1500 < last <= 1500 + 1500 + 40


def test_two_walkers_keep_their_own_ids():
    follow = primed()
    a = lambda ms: where(ms)  # noqa: E731
    b = lambda ms: (0.8 - 0.0001 * ms, 0.2, 0.9 - 0.0001 * ms, 0.6)  # noqa: E731
    out = run(follow, 2500, ours=lambda ms: {"person": [a(ms) + (0.9,), b(ms) + (0.8,)]},
              protect=lambda at: [theirs(at, "A", box=a(at)), theirs(at, "B", box=b(at))])
    for p in out:
        ms = p["clockWall"] + STAMP - WALL0
        for tr in p["tracks"]:
            want = a(ms) if tr["id"] == "A" else b(ms)
            assert abs(to_box(tr["coord"])[0] - want[0]) < 0.005
    assert {tr["id"] for p in out for tr in p["tracks"]} == {"A", "B"}


def test_faces_and_other_kinds():
    follow = primed()
    face = lambda ms: (0.14 + 0.0002 * ms, 0.31, 0.17 + 0.0002 * ms, 0.37)  # noqa: E731
    out = run(follow, 2000, ours=lambda ms: {"person": [where(ms) + (0.9,)], "face": [face(ms) + (0.8,)]},
              protect=lambda at: [theirs(at), theirs(at, "f1", "face", box=face(at)), theirs(at, "lp", "licensePlate", box=(0.5, 0.5, 0.52, 0.51))])
    ids = {tr["id"]: tr["objectType"] for p in out for tr in p["tracks"]}
    assert ids == {"p1": "person", "f1": "face"}  # a plate is Protect's alone


def test_followed_before_protect_says_anything():
    """With nothing reported yet nothing goes out, but the walker is followed, and the first report binds it:
    we had it this much earlier."""
    follow = CameraFollow(STAMP)
    assert run(follow, 2000) == [] and len(follow.tracker.tracks) == 1
    out = run(follow, 4000, start=2000, protect=lambda at: [theirs(at)] if at >= 1500 else None)
    assert out and follow.stats["bound"] == 1
    head = follow.head_start()["person"]  # (a person counts wherever they were first seen)
    assert head["n"] == 1 and 1400 <= head["frames_ms"] <= 1600  # seen from 0; Protect's first report is of the frame at 1500
    assert 1900 <= head["told_ms"] <= 2100  # ... and reached us half a second after that
    for p in out:
        ms = p["clockWall"] + STAMP - WALL0
        assert abs(to_box(p["tracks"][0]["coord"])[0] - where(ms)[0]) < 0.004


def test_the_face_of_a_followed_person_goes_out_too():
    follow = primed()
    face = lambda ms: (0.13 + 0.0002 * ms, 0.31, 0.17 + 0.0002 * ms, 0.37)  # noqa: E731
    out = run(follow, 2000, ours=lambda ms: {"person": [where(ms) + (0.9,)], "face": [face(ms) + (0.8,)]}, protect=lambda at: [theirs(at)])
    faces = [tr for p in out for tr in p["tracks"] if tr["objectType"] == "face"]
    assert len(faces) > 30 and all(tr["id"] == "p1:face" and tr["of"] == "p1" and tr["local"] for tr in faces)
    # a face on its own, with nobody followed, is not sent
    lone = primed()
    assert run(lone, 2000, ours=lambda ms: {"face": [face(ms) + (0.8,)]}, protect=lambda at: []) == []


def test_people_protect_never_reported_are_counted():
    follow = primed()
    run(follow, 3000, ours=lambda ms: where(ms) if ms < 2000 else None, protect=lambda at: [])
    assert follow.stats["people_never_reported"] == 1 and follow.stats["bound"] == 0


def test_busy_while_something_moves():
    follow = primed()
    car = lambda ms: {"vehicle": [(0.2 + 0.001 * ((ms // 33) % 2), 0.2, 0.4, 0.4, 0.5)] if (ms // 400) % 2 else []}  # noqa: E731
    run(follow, 3000, ours=car)  # a parked car, its box trembling, lost and found again
    assert not follow.busy()
    follow = primed()
    run(follow, 1000, ours=lambda ms: (0.5, 0.5, 0.6, 0.9))  # someone appears, and stands still
    assert follow.busy()  # new: worth a close look for a moment
    run(follow, 4000, start=1000, ours=lambda ms: (0.5, 0.5, 0.6, 0.9))
    assert not follow.busy()
    run(follow, 5000, start=4000, ours=lambda ms: (0.5 + 0.0002 * (ms - 4000), 0.5, 0.6 + 0.0002 * (ms - 4000), 0.9))  # walks off
    assert follow.busy()
    run(follow, 8500, start=5000, ours=lambda ms: None, protect=lambda at: [theirs(at, "v", "vehicle", stationary=True, box=(0.1, 0.1, 0.2, 0.2))])
    assert not follow.busy()  # Protect's parked car is no reason
    follow.report(report(8600, [theirs(8600, "x")]), WALL0 + 9000)
    assert follow.busy()  # Protect says something moves


def test_tracker_follows_through_a_crossing():
    tracker = LocalTracker()
    ids = {}
    for i in range(60):
        ms = i * 33.3
        a = (0.1 + 0.0003 * ms, 0.3, 0.2 + 0.0003 * ms, 0.7, 0.9)
        b = (0.7 - 0.0003 * ms, 0.32, 0.8 - 0.0003 * ms, 0.72, 0.9)
        seen = tracker.update(ms, {"person": [a, b]})
        for t in seen:
            ids.setdefault(t.id, []).append(t.box[0])
    assert len(ids) == 2  # no new tracks where they pass each other
    left, right = sorted(ids.values(), key=lambda xs: xs[0])
    assert left[-1] > left[0] and right[-1] < right[0]  # each kept going its own way


def test_a_parked_thing_protect_calls_moving_is_not_taken_up():
    follow = primed()
    still = lambda ms: (0.5 + 0.001 * ((ms // 33) % 2), 0.5, 0.7, 0.7)  # noqa: E731 - its box trembles, no more
    out = run(follow, 3000, ours=lambda ms: {"vehicle": [still(ms) + (0.9,)]},
              protect=lambda at: [theirs(at, "bike", "vehicle", stationary=(at % 1000 != 0), box=still(at))])
    assert out == [] and follow.stats["bound"] == 0


def test_protects_word_and_ours_are_separate():
    follow = primed()
    out = run(follow, 2000, protect=lambda at: [theirs(at)])
    tr = out[-1]["tracks"][0]
    assert tr["stationary"] is False and tr["moving"] is True
    # it stops; Protect has not said stationary yet: still followed, and we say it is not moving
    halt = lambda ms: where(min(ms, 2000))  # noqa: E731
    out = run(follow, 4000, start=2000, ours=halt, protect=lambda at: [theirs(at, box=halt(at))])
    tr = out[-1]["tracks"][0]
    assert tr["stationary"] is False and tr["moving"] is False


PARKED_BOX = (0.45, 0.4, 0.65, 0.6)


def passing(ms: float) -> dict:
    """A car drives left to right past a parked one of the same size. The parked one's box comes and goes (as
    a detector's does); while the two overlap only one box is found, the parked car's."""
    x = 0.05 + 0.0002 * ms
    mover = (x, 0.42, x + 0.2, 0.62, 0.9)
    overlap = min(mover[2], PARKED_BOX[2]) - max(mover[0], PARKED_BOX[0])
    parked_seen = ms < 600 or ms > 1500  # not found for a while just before the car reaches it
    boxes = []
    if overlap > 0.1:  # one behind the other: one box, where the parked car is
        boxes.append(PARKED_BOX + (0.8,))
    else:
        boxes.append(mover)
        if parked_seen:
            boxes.append(PARKED_BOX + (0.7,))
    return {"vehicle": boxes}


def test_a_passing_car_does_not_hand_its_track_to_a_parked_one():
    follow = primed()
    mover = lambda at: (0.05 + 0.0002 * at, 0.42, 0.25 + 0.0002 * at, 0.62)  # noqa: E731
    protect = lambda at: [theirs(at, "car", "vehicle", box=mover(at)),  # noqa: E731
                          theirs(at, "kerb", "vehicle", stationary=True, box=PARKED_BOX)]
    out = run(follow, 4500, ours=passing, protect=protect)
    wrong = []
    for p in out:
        ms = p["clockWall"] + STAMP - WALL0
        for tr in p["tracks"]:
            assert tr["id"] == "car"  # the parked one is Protect's alone
            if abs(to_box(tr["coord"])[0] - mover(ms)[0]) > 0.06:
                wrong.append(round(ms))
    assert not wrong, f"the passing car's id sat somewhere else (on the parked car) at {wrong[:5]} ... {len(wrong)} frames"
    assert max(p["clockWall"] + STAMP - WALL0 for p in out) > 4000  # and it is followed again once past


def test_the_tracker_remembers_where_things_stand_by_itself():
    """No word from Protect: a car has stood there for seconds, is not found for a moment, and just then
    another drives over the spot. The passing car's track must not end up on the parked one."""
    tracker = LocalTracker()
    mover_id = None
    for i in range(330):
        ms = i * 100 / 3
        x = -1.0 + 0.0002 * ms  # the passing car comes into view at 5 s
        mover = (x, 0.42, x + 0.2, 0.62, 0.9)
        overlap = min(mover[2], PARKED_BOX[2]) - max(mover[0], PARKED_BOX[0])
        boxes = []
        if x > 0 and overlap <= 0.1:
            boxes.append(mover)
        if ms < 6000 or overlap > 0.1 or ms > 8500:  # found, then not, then found again as the other passes
            boxes.append(PARKED_BOX + (0.8,))
        for t in tracker.update(ms, {"vehicle": boxes}):
            if mover_id is None and x > 0.02 and abs(t.box[0] - x) < 0.02:
                mover_id = t.id
    assert mover_id is not None and tracker.places
    track = tracker.tracks.get(mover_id)
    assert track is None or abs(track.box[0] - PARKED_BOX[0]) > 0.1, "the passing car's track is sitting on the parked car"


def test_a_slow_thing_far_off_is_not_broken_up_by_its_own_place():
    """A car far off creeps along, stops for five seconds (long enough to count as standing there), and goes
    on. It is one track throughout: the place it stood in is its own."""
    tracker = LocalTracker()
    ids = set()
    for i in range(450):
        ms = i * 100 / 3
        x = 0.3 + 0.00002 * min(ms, 4000) + 0.00002 * max(0, ms - 9000)
        for t in tracker.update(ms, {"vehicle": [(x, 0.3, x + 0.05, 0.34, 0.8)]}):
            ids.add(t.id)
    assert tracker.places and len(ids) == 1


def test_a_small_thing_far_off_found_only_now_and_then_stays_one_track():
    tracker = LocalTracker()
    ids = set()
    for i in range(300):
        ms = i * 100 / 3
        x = 0.5 + 0.00001 * ms
        found = [(x, 0.1, x + 0.04, 0.13, 0.4)] if (ms // 400) % 3 == 0 else []  # found for 0.4 s, not for 0.8 s
        for t in tracker.update(ms, {"vehicle": found}):
            ids.add(t.id)
    assert len(ids) == 1


def test_small_and_far_off_is_left_to_protect_until_it_is_nearer():
    follow = primed()
    # a car comes up the road: small at first, growing
    size = lambda ms: 0.03 + 0.00003 * ms  # noqa: E731 - its height: 3% of the picture, 6% at one second, 15% at four
    car = lambda ms: (0.3 + 0.0001 * ms, 0.2, 0.3 + 0.0001 * ms + 2 * size(ms), 0.2 + size(ms))  # noqa: E731
    out = run(follow, 4000, ours=lambda ms: {"vehicle": [car(ms) + (0.9,)]}, protect=lambda at: [theirs(at, "car", "vehicle", box=car(at))])
    first = min(p["clockWall"] + STAMP - WALL0 for p in out)
    assert 1000 <= first <= 1400 and follow.stats["bound"] == 1  # taken up once it is big enough to hold on to
    # and one driving away is handed back when it has gone small
    follow = primed()
    away = lambda ms: car(4000 - ms)  # noqa: E731
    out = run(follow, 4600, ours=lambda ms: {"vehicle": [away(ms) + (0.9,)]}, protect=lambda at: [theirs(at, "car", "vehicle", box=away(at))])
    last = max(p["clockWall"] + STAMP - WALL0 for p in out)
    # (Protect's report of the frame in which it has gone small comes half a second after that frame)
    assert follow.stats["let_go"]["far"] == 1 and follow.stats["bound"] == 1 and 3400 <= last <= 4300
