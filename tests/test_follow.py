"""Protect's tracks followed in our frames: binding, sending, letting go."""

from tracker_service.follow import CameraFollow, restamp, to_box
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
    times = [p["clockWall"] - WALL0 for p in out]
    assert times == sorted(times) and len(set(times)) == len(times)  # in order, each frame once
    # from the frame of the first report on, every frame has a box: about 30 a second
    assert min(times) < 50 and max(times) > 2900 and len(times) >= 85
    for p in out:
        (tr,) = p["tracks"]
        assert tr["id"] == "p1" and tr["objectType"] == "person" and tr["zones"] == [3] and tr["local"] is True
        assert p["stampMs"] == 0 and p["source"] == "local" and p["eventId"] == "ev1"
        assert abs(p["clockStream"] - (p["clockWall"] - WALL0 + STREAM0)) < 0.2
        # the box is where the walker was in that very frame
        ms = p["clockWall"] - WALL0
        assert abs(to_box(tr["coord"])[0] - where(ms)[0]) < 0.004


def test_the_first_boxes_are_sent_when_the_binding_is_made():
    follow = primed()
    out = run(follow, 560, protect=lambda at: [theirs(at)])  # the first report (of the frame at 0) comes at 500
    assert len(out) >= 14 and out[0]["clockWall"] - WALL0 < 50  # the half second already seen, at once


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
    last = max(p["clockWall"] - WALL0 for p in out)
    assert 2000 < last < 3000  # three reports that disagree (each half a second late), then nothing more


def test_stops_when_the_local_track_is_lost_and_binds_again():
    follow = primed()
    ours = lambda ms: where(ms) if ms < 1000 or ms > 2000 else None  # noqa: E731 - hidden for a second
    out = run(follow, 4000, ours=ours, protect=lambda at: [theirs(at)])
    times = [p["clockWall"] - WALL0 for p in out]
    assert not [t for t in times if 1000 + 320 < t < 2000]  # it coasts a moment, then nothing
    assert [t for t in times if 1000 <= t <= 1000 + 320]
    assert max(times) > 3900 and follow.stats["bound"] == 2


def test_stops_when_protect_stops_reporting():
    follow = primed()
    out = run(follow, 5000, protect=lambda at: [theirs(at)] if at <= 1500 else None)
    last = max(p["clockWall"] - WALL0 for p in out)
    assert 1500 < last <= 1500 + 1500 + 40


def test_two_walkers_keep_their_own_ids():
    follow = primed()
    a = lambda ms: where(ms)  # noqa: E731
    b = lambda ms: (0.8 - 0.0001 * ms, 0.2, 0.9 - 0.0001 * ms, 0.6)  # noqa: E731
    out = run(follow, 2500, ours=lambda ms: {"person": [a(ms) + (0.9,), b(ms) + (0.8,)]},
              protect=lambda at: [theirs(at, "A", box=a(at)), theirs(at, "B", box=b(at))])
    for p in out:
        ms = p["clockWall"] - WALL0
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


def test_frames_before_the_clocks_are_known_are_skipped():
    follow = CameraFollow(STAMP)
    assert follow.frame(rtp_of(0), {"person": [where(0) + (0.9,)]}, WALL0) == []
    assert follow.stats["untimed"] == 1


def test_restamp():
    got = restamp({"clockWall": 1000.0, "clockStream": 90_000, "clockStreamRate": 90_000, "tracks": [1]}, -125)
    assert got["clockWall"] == 875 and got["clockStream"] == 90_000 - 11_250 and got["stampMs"] == 0 and got["source"] == "protect" and got["tracks"] == [1]


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
