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
    every = play(follow, 9000, lambda ms: where(ms) if ms < 2000 else None)
    out = [p for p in every if p["tracks"]]
    last = max(frame_ms(p) for p in out)
    assert 1950 <= last <= 2000 + 1540  # shown where they should be for a second and a half, then nothing
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
    assert out and 3000 < frame_ms(out[0]) < 3500 and out[0]["tracks"][0]["objectType"] == "vehicle"


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


def test_a_parked_car_whose_box_shifts_when_something_passes_is_not_told():
    """What the live stream showed: a parked car's box is pushed about (half hidden, found narrower, found
    again) each time a car passes in front, and lost in between. It is never something to follow."""
    follow = CameraFollow(STAMP, epoch="p1")
    def scene(ms):
        phase = (ms // 1500) % 3
        if phase == 1:
            return {"vehicle": []}  # hidden
        shove = 0.03 if phase == 2 and (ms % 1500) < 400 else 0.0  # found again a little to the side, then where it is
        return {"vehicle": [(0.05 + shove, 0.2, 0.2 + shove * 0.3, 0.33, 0.87)]}
    out = play(follow, 12000, scene, lambda at: [theirs(at, "kerb", "vehicle", stationary=True, box=(0.05, 0.2, 0.2, 0.33))])
    assert [p for p in out if p["tracks"]] == [] and follow.local.stats["told"] == 0


def test_a_car_that_parks_is_left_out_again_and_comes_back_when_it_drives_off():
    follow = CameraFollow(STAMP, epoch="p2")
    x = lambda ms: 0.1 + 0.0002 * min(ms, 2000) + 0.0002 * max(0, ms - 12000)  # noqa: E731 - drives in, stands ten seconds, drives on
    out = play(follow, 14000, lambda ms: {"vehicle": [(x(ms), 0.3, x(ms) + 0.15, 0.42, 0.9)]})
    told = sorted(frame_ms(p) for p in out if p["tracks"])
    assert told[0] < 600  # told as it drives in
    parked = [t for t in told if 7500 < t < 11900]
    assert not parked  # some seconds after it stopped it is no longer told ...
    assert [t for t in told if t > 12600]  # ... until it drives off again
    assert {tr["id"] for p in out for tr in p["tracks"]} == {"Lp2-1"}  # the same track all along


def test_a_person_standing_still_stays_told():
    follow = CameraFollow(STAMP, epoch="p3")
    out = play(follow, 12000, lambda ms: where(min(ms, 1500)))
    told = [frame_ms(p) for p in out if p["tracks"]]
    assert told and max(told) > 11900 and out[-1]["tracks"][0]["moving"] is False


def test_a_car_driving_off_into_the_distance_is_let_go_when_it_is_small():
    follow = CameraFollow(STAMP, epoch="d1")
    h = lambda ms: max(0.01, 0.16 - 0.00004 * ms)  # noqa: E731 - its height: 16% of the picture, 4% after three seconds
    car = lambda ms: (0.2 + 0.0001 * ms, 0.3, 0.2 + 0.0001 * ms + 2 * h(ms), 0.3 + h(ms))  # noqa: E731
    out = play(follow, 6000, lambda ms: {"vehicle": [car(ms) + (0.9,)]})
    told = [frame_ms(p) for p in out if p["tracks"]]
    assert told and min(told) < 1000 and 2800 <= max(told) <= 3200  # until it is under about 4% of the picture's height


def test_every_person_counts_for_the_head_start():
    follow = CameraFollow(STAMP, epoch="h1")
    play(follow, 4000, where, lambda at: [theirs(at)] if at >= 1500 else None)
    head = follow.head_start()["person"]
    assert head["n"] == 1 and 1400 <= head["frames_ms"] <= 1600


def test_one_person_up_the_path_is_one_track_from_entry_to_exit():
    """The core case for a camera on a porch: someone walks up the path behind two posts (hidden for 0.8 s and
    1.6 s), is half hidden by a railing (for a while the top half of them is found as well as, or in place
    of, the whole), stoops to put a parcel down, turns round and walks back out the way they came."""
    def person(ms: float):
        if ms < 6000:
            x = 0.05 + 0.0001 * ms  # walking in
        elif ms < 9000:
            x = 0.65  # at the door
        else:
            x = 0.65 - 0.0001 * (ms - 9000)  # and back
        top, bottom = 0.3, 0.7
        if 7000 <= ms < 8200:
            top = 0.5  # bent over the parcel
        return (x, top, x + 0.1, bottom)

    def scene(ms: float) -> dict:
        if ms >= 15200:
            return {"person": []}  # out of the picture
        if 1500 <= ms < 2300 or 3800 <= ms < 5400 or 11000 <= ms < 12200:
            return {"person": []}  # behind a post
        x1, y1, x2, y2 = person(ms)
        boxes = [(x1, y1, x2, y2, 0.88)]
        if 2600 <= ms < 3400:  # behind the railing: the top half found as a person too
            boxes.append((x1 + 0.005, y1, x2 - 0.005, y1 + 0.22, 0.7))
        if 12600 <= ms < 13400:  # ... or only the top half
            boxes = [(x1, y1, x2, y1 + 0.2, 0.6)]
        return {"person": boxes}

    follow = CameraFollow(STAMP, epoch="path")
    protect = lambda at: [theirs(at, "P1", box=person(at))] if 2500 <= at <= 10000 else []  # noqa: E731
    out = play(follow, 16500, scene, protect)
    told = [p for p in out if p["tracks"]]
    ids = {tr["id"] for p in told for tr in p["tracks"]}
    assert ids == {"Lpath-1"}, f"one person became {len(ids)} tracks: {sorted(ids)}"
    assert all(len(p["tracks"]) == 1 for p in told)  # never two at once
    times = [frame_ms(p) for p in told]
    assert min(times) < 800 and 15000 < max(times) < 15900  # from coming in to going out, and gone soon after
    assert follow.local.stats == {"told": 1, "confirmed": 1, "people": 1}
    # Protect's report attached to that one track and stayed on it
    assert told[-1]["tracks"][0]["protectId"] == "P1" and told[-1]["tracks"][0]["confirmed"] is True
    head = follow.head_start()["person"]
    assert head["n"] == 1 and head["frames_ms"] >= 2000  # we had them two seconds and more before Protect's first report


def test_two_people_passing_stay_two():
    follow = CameraFollow(STAMP, epoch="two")
    a = lambda ms: (0.1 + 0.0001 * ms, 0.3, 0.2 + 0.0001 * ms, 0.7, 0.9)  # noqa: E731
    b = lambda ms: (0.8 - 0.0001 * ms, 0.32, 0.9 - 0.0001 * ms, 0.72, 0.9)  # noqa: E731
    out = play(follow, 6000, lambda ms: {"person": [a(ms), b(ms)] if not 2800 < ms < 3600 else [a(ms)]})  # one hides the other as they pass
    told = [p for p in out if p["tracks"]]
    assert {tr["id"] for p in told for tr in p["tracks"]} == {"Ltwo-1", "Ltwo-2"}
    last = {tr["id"]: tr["coord"][0] / 1000 for tr in told[-1]["tracks"]}
    assert last["Ltwo-1"] > 0.6 and last["Ltwo-2"] < 0.3  # each went on its own way


def test_someone_who_walks_out_of_the_picture_is_gone_at_once():
    follow = CameraFollow(STAMP, epoch="out")
    out = play(follow, 6000, lambda ms: (0.7 + 0.0002 * ms, 0.3, 0.8 + 0.0002 * ms, 0.7) if ms < 1000 else None)
    told = [frame_ms(p) for p in out if p["tracks"]]
    assert told and max(told) < 1000 + 600 and follow.local.tracks == {}
