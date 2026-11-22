"""The stage 0 measurement finds a timing that was put there."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))

from tools.spike import analyse  # noqa: E402
from tracker_service.rtsp import plain  # noqa: E402


def made_up(stamp_ms: float, wraps: bool = False) -> dict:
    """A person walking across at an even pace: our box in every frame (30 a
    second), Protect's four times a second, stamped `stamp_ms` after the
    frame they describe."""
    first_rtp = (1 << 32) - 90 * 2000 if wraps else 90 * 5_000_000
    stream0 = first_rtp / 90 + (3 * (1 << 32) / 90)  # the stream clock has been round three times
    where = lambda ms: 0.1 + 0.00012 * ms  # noqa: E731 - left edge at a moment (ms after the start)
    frames, reports = [], []
    for i in range(300):
        ms = i * 1000 / 30
        x = where(ms)
        frames.append({"rtp": int(first_rtp + ms * 90) % (1 << 32), "at": 1_000_000 + ms + 80, "done": 1_000_000 + ms + 100,
                       "boxes": {"person": [(x, 0.3, x + 0.1, 0.7, 0.9)], "vehicle": [(0.6, 0.6, 0.9, 0.9, 0.9)]}})
    for k in range(4, 36):
        ms = k * 250
        x = where(ms)
        reports.append({"clockStream": stream0 + ms + stamp_ms, "clockStreamRate": 1000, "clockWall": 1_700_000_000_000 + ms + stamp_ms, "got": 1_000_000 + ms + 500,
                        "tracks": [{"id": "1", "objectType": "person", "coord": [x * 1000, 300, 100, 400], "stationary": False},
                                   {"id": "2", "objectType": "vehicle", "coord": [600, 600, 300, 300], "stationary": True}]})
    return {"frames": frames, "reports": reports, "times": {}, "size": [1280, 720], "dropped": 0}


def test_finds_the_stamp():
    for stamp in (125, 25, 300):
        got = analyse(made_up(stamp), out=lambda *_: None)["kinds"]["person"]
        assert abs(got["best_ms"] + stamp) <= 20, (stamp, got)
        assert got["overlap"] > 0.9
        assert "vehicle" not in analyse(made_up(stamp), out=lambda *_: None)["kinds"]  # parked: says nothing of timing


def test_across_the_clock_starting_over():
    got = analyse(made_up(125, wraps=True), out=lambda *_: None)["kinds"]["person"]
    assert abs(got["best_ms"] + 125) <= 20


def test_srtp_is_not_asked_for():
    assert plain("rtsps://host:7441/KEY?enableSrtp") == "rtsps://host:7441/KEY"
    assert plain("rtsps://host:7441/KEY?enableSrtp&x=1") == "rtsps://host:7441/KEY?x=1"
    assert plain("rtsp://host/stream") == "rtsp://host/stream"
