"""On the engine's own word: told without Protect, Protect's details added in flight."""

from test_follow import STAMP, STREAM0, WALL0, report, rtp_of, theirs, where

from tracker_service.follow import CameraFollow, to_box

STEP = 100 / 3


def play(follow: CameraFollow, until: float, ours, protect=None, start: float = 0.0) -> list[dict]:
    """Frames every 33 ms; Protect's reports (of the frame half a second before) every 250 ms. Returns the local packets."""
    out = []
    for i in range(int(start / STEP), int(until / STEP)):
        ms = i * STEP
        due = ms - 500
        if protect and due >= 0 and due // 250 != (due - STEP) // 250:
            tracks = protect((due // 250) * 250)
            if tracks is not None:
                follow.report(report((due // 250) * 250, tracks), WALL0 + ms)
        found = ours(ms)
        follow.frame(rtp_of(ms), found if isinstance(found, dict) else {"person": [found + (0.9,)]} if found else {}, WALL0 + ms)
        out += follow.local_packets
    return out


def frame_ms(packet: dict) -> float:
    """Which frame a packet is of, from its stream clock (the clock a card matches the picture with)."""
    return packet["clockStream"] + STAMP - STREAM0


def test_a_person_is_told_with_no_word_from_protect():
    follow = CameraFollow(STAMP, epoch="t1")
    out = play(follow, 3000, where)
    assert follow.frame(rtp_of(3000), {}, WALL0 + 3000) == []  # in Protect's name: nothing
    out = [p for p in out if p["tracks"]]
    first = frame_ms(out[0])
    assert 250 <= first <= 450  # after a quarter of a second, once it has also moved a little
    times = [frame_ms(p) for p in out]
    assert len(times) >= 75 and times == sorted(times)  # every frame from then on
    for p in out:
        (tr,) = p["tracks"]
        assert tr["id"] == "Lt1-1" and tr["objectType"] == "person" and tr["local"] is True and tr["confirmed"] is False
        assert tr["protectId"] is None and tr["eventId"] is None and tr["zones"] == [] and tr["desc"] == "" and tr["stationary"] is False
        assert p["source"] == "local" and p["mode"] == "local" and p["snapshot"] is True and tr["moving"] is True
        assert abs(to_box(tr["coord"])[0] - where(frame_ms(p))[0]) < 0.004  # the box of that very frame
        assert p["clockStreamRate"] == 1000


def test_protects_details_are_added_in_flight_and_stay():
    follow = CameraFollow(STAMP, epoch="t2")
    protect = lambda at: [theirs(at, "P9", desc="tall")] if 1000 <= at <= 2500 else []  # noqa: E731
    every = play(follow, 5000, where, protect)
    out = [p for p in every if p["tracks"]]
    ids = {tr["id"] for p in out for tr in p["tracks"]}
    assert ids == {"Lt2-1"}  # one track from start to end, whatever Protect does
    before = [p for p in out if not p["tracks"][0]["confirmed"]]
    after = [p for p in out if p["tracks"][0]["confirmed"]]
    assert before and after and max(frame_ms(p) for p in before) < min(frame_ms(p) for p in after)
    assert 1400 <= min(frame_ms(p) for p in after) <= 1700  # Protect's first report (of the frame at 1000) comes at 1500
    tr = after[0]["tracks"][0]
    assert tr["protectId"] == "P9" and tr["desc"] == "tall" and tr["zones"] == [3] and tr["confidence"] == 80 and tr["eventId"] == "ev1"
    last = out[-1]["tracks"][0]  # Protect has long stopped reporting it: still told, its details kept
    assert frame_ms(out[-1]) > 4900 and last["confirmed"] is True and last["protectId"] == "P9" and last["desc"] == "tall"
    assert follow.local.stats == {"told": 1, "confirmed": 1, "people": 1}
    # (the stream clock is the same one before and after Protect spoke: no jump in what the picture is matched with)
    steps = [b - a for a, b in zip([frame_ms(p) for p in out], [frame_ms(p) for p in out][1:])]
    assert max(steps) < 40
    # the wall clock was ours (here: half a second off Protect's) and is brought to Protect's without a jump or a step back
    for a, b in zip(every, every[1:]):
        d_stream, d_wall = b["clockStream"] - a["clockStream"], b["clockWall"] - a["clockWall"]
        assert d_stream > 0 and d_wall > 0 and abs(d_wall - d_stream) <= 0.1 * d_stream + 0.2


def test_ends_cleanly_when_it_leaves():
    follow = CameraFollow(STAMP, epoch="t3")
    every = play(follow, 6000, lambda ms: where(ms) if ms < 2000 else None)
    out = [p for p in every if p["tracks"]]
    last = max(frame_ms(p) for p in out)
    assert 1950 <= last <= 2000 + 540  # it coasts until the tracker gives it up, then nothing
    assert follow.local.tracks == {}
    # each packet is the whole of what is told: the first one after it has gone is empty, and then one a second says "nothing"
    after = [frame_ms(p) for p in every if frame_ms(p) > last]
    assert after and after[0] - last < 40 and all(not p["tracks"] for p in every if frame_ms(p) > last)
    gaps = [b - a for a, b in zip(after, after[1:])]
    assert gaps and all(950 <= g <= 1050 for g in gaps)


def test_what_only_stands_there_is_never_told():
    follow = CameraFollow(STAMP, epoch="t4")
    car = (0.2, 0.2, 0.4, 0.4)
    bike = lambda ms: (0.6 + 0.002 * ((ms // 33) % 2), 0.5, 0.7, 0.7)  # noqa: E731 - its box trembles
    scene = lambda ms: {"vehicle": [car + (0.9,)] + ([bike(ms) + (0.8,)] if (ms // 500) % 2 else []), "person": []}  # noqa: E731
    out = play(follow, 6000, scene, lambda at: [theirs(at, "v1", "vehicle", stationary=True, box=car)])
    assert [p for p in out if p["tracks"]] == [] and follow.local.stats["told"] == 0


def test_a_parked_car_that_drives_off_is_told_from_when_it_moves():
    follow = CameraFollow(STAMP, epoch="t5")
    car = lambda ms: (0.2 + 0.0003 * max(0, ms - 3000), 0.2, 0.4 + 0.0003 * max(0, ms - 3000), 0.4)  # noqa: E731
    out = [p for p in play(follow, 5000, lambda ms: {"vehicle": [car(ms) + (0.9,)]}) if p["tracks"]]
    assert out and 3000 < frame_ms(out[0]) < 3300 and out[0]["tracks"][0]["objectType"] == "vehicle"


def test_too_small_or_too_unsure_is_not_told():
    follow = CameraFollow(STAMP, epoch="t6")
    tiny = lambda ms: {"person": [(0.1 + 0.0002 * ms, 0.5, 0.11 + 0.0002 * ms, 0.52, 0.9)]}  # noqa: E731
    unsure = lambda ms: {"person": [where(ms) + (0.35,)]}  # noqa: E731
    assert [p for p in play(follow, 2000, tiny) if p["tracks"]] == []
    assert [p for p in play(CameraFollow(STAMP, epoch="t7"), 2000, unsure) if p["tracks"]] == []


def test_two_people_and_a_face():
    follow = CameraFollow(STAMP, epoch="t8")
    other = lambda ms: (0.8 - 0.0002 * ms, 0.3, 0.9 - 0.0002 * ms, 0.7)  # noqa: E731
    face = lambda ms: (0.13 + 0.0002 * ms, 0.31, 0.17 + 0.0002 * ms, 0.37)  # noqa: E731
    out = play(follow, 2000, lambda ms: {"person": [where(ms) + (0.9,), other(ms) + (0.8,)], "face": [face(ms) + (0.8,)]})
    last = {tr["id"]: tr for tr in out[-1]["tracks"]}
    assert set(last) == {"Lt8-1", "Lt8-2", "Lt8-1:face"}
    assert last["Lt8-1:face"]["of"] == "Lt8-1" and last["Lt8-1:face"]["objectType"] == "face"


def test_the_careful_half_is_unchanged_beside_it():
    follow = CameraFollow(STAMP, epoch="t9")
    hybrid = []
    for i in range(90):
        ms = i * STEP
        due = ms - 500
        if due >= 0 and due // 250 != (due - STEP) // 250:
            hybrid += follow.report(report((due // 250) * 250, [theirs((due // 250) * 250)]), WALL0 + ms)
        hybrid += follow.frame(rtp_of(ms), {"person": [where(ms) + (0.9,)]}, WALL0 + ms)
    assert hybrid and {tr["id"] for p in hybrid for tr in p["tracks"]} == {"p1"}  # Protect's id there, ours here
    assert follow.local_packets[0]["tracks"][0]["id"] == "Lt9-1" and follow.local_packets[0]["tracks"][0]["protectId"] == "p1"


def test_the_wall_clock_is_brought_to_protects_gently():
    """Our own idea of the wall clock is 400 ms off. Protect speaks at 2 s: no jump, no step back, and four
    seconds later the two agree; the stream clock (what the picture is matched with) is never touched."""
    follow = CameraFollow(STAMP, epoch="s1")
    out = []
    for i in range(240):
        ms = i * STEP
        if i == 60:
            follow.report(report(ms - 400, []), WALL0 + ms + 400)
        follow.frame(rtp_of(ms), {"person": [where(ms) + (0.9,)]}, WALL0 + ms + 400)  # our clock: 400 ms ahead
        out += follow.local_packets
    off = [p["clockWall"] - (WALL0 + frame_ms(p) - STAMP) for p in out]  # how far the packet's wall clock is from Protect's
    assert abs(off[0] - 400) < 1 and abs(off[-1]) < 1
    for a, b in zip(out, out[1:]):
        d_stream, d_wall = b["clockStream"] - a["clockStream"], b["clockWall"] - a["clockWall"]
        assert d_wall > 0 and abs(d_wall - d_stream) <= 0.1 * d_stream + 0.2
