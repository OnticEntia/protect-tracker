"""The replay tool's reading of what the tracker did, on a made-up visit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))

from tools.replay import replay  # noqa: E402
from tracker_service import tracks  # noqa: E402


def visit():
    """Someone walks in, is hidden behind a post for 1.5 s, stoops, and walks on out of the picture."""
    for i in range(360):
        ms = i * 100 / 3
        x = 0.1 + 0.00007 * ms
        if 3000 <= ms < 4500:
            yield ms, {"person": []}
        elif 6000 <= ms < 7000:
            yield ms, {"person": [(x, 0.5, x + 0.1, 0.7, 0.85)]}  # bent over
        elif ms < 11000:
            yield ms, {"person": [(x, 0.3, x + 0.1, 0.7, 0.9)], "face": [(x + 0.03, 0.31, x + 0.07, 0.37, 0.8)]}
        else:
            yield ms, {"person": []}


def test_one_person_one_track_now_and_several_before():
    looks = list(visit())
    now = replay(looks)
    assert len(now["tracks"]) == 1 and now["breaks"] == [] and now["most_at_once"] == 1
    assert now["people_frames"] > 250 and now["face_frames"] > 100 and now["face_tracks"] == 1
    before = replay(looks, old=True)
    assert len(before["tracks"]) >= 2
    assert any(b["why"].startswith("hidden") and b["gap_ms"] > 1000 for b in before["breaks"])
    assert tracks.GONE_PERSON_MS == 3000  # the old rules were only borrowed
