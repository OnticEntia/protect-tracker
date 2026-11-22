from tracker_service.clock import RTP_WRAP_MS, FeedClock


def test_frame_time_on_the_cameras_clock():
    clock = FeedClock()
    assert clock.frame_wall(0, 0) is None  # nothing reported yet
    # a report: the wall clock is 1 700 000 000 000 when the stream clock is 50 000 ms
    clock.report(1_700_000_000_000, 50_000, 1000, now_ms=10_000)
    rtp = (50_000 - 200) * 90  # a frame stamped 200 ms before that
    assert clock.frame_wall(rtp, now_ms=10_050) == 1_700_000_000_000 - 200


def test_stream_clock_in_other_units():
    clock = FeedClock()
    clock.report(1000, 90_000 * 50, 90_000, now_ms=0)
    assert clock.frame_wall(90 * 49_900, now_ms=0) == 900


def test_the_rtp_clock_starts_over():
    clock = FeedClock()
    stream = 3 * RTP_WRAP_MS + 5_000  # the stream clock has been round three times
    clock.report(2_000_000, stream, 1000, now_ms=0)
    rtp = round((5_000 - 100) * 90)  # the frame's 32-bit timestamp knows nothing of the rounds
    assert abs(clock.frame_wall(rtp, now_ms=0) - (2_000_000 - 100)) < 1


def test_another_clock_is_refused():
    clock = FeedClock()
    clock.report(2_000_000, 50_000, 1000, now_ms=0)
    assert clock.frame_wall(90 * 80_000, now_ms=0) is None  # half a minute ahead: not this clock
    assert clock.frame_wall(90 * 20_000, now_ms=0) is None  # half a minute behind
