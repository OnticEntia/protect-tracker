"""The model comparison's arithmetic."""

from tracker_service.bench import Bench, band, match, score

NEAR = (0.1, 0.2, 0.3, 0.6, 0.9)  # a person near (tall)
FAR = (0.6, 0.30, 0.62, 0.34, 0.8)  # one far off (small)
CAR = (0.4, 0.5, 0.6, 0.6, 0.9)


def test_bands_and_matching():
    assert band(NEAR) == "large" and band(FAR) == "small" and band(CAR) == "medium"
    assert match([NEAR, FAR], [FAR, NEAR]) == [(0, 1, 1.0), (1, 0, 1.0)]
    assert match([NEAR], [CAR]) == []


def test_a_model_that_misses_what_is_far_off():
    reference = [{"person": [NEAR, FAR], "vehicle": [CAR]}] * 10
    candidate = [{"person": [(0.11, 0.2, 0.3, 0.6, 0.9)], "vehicle": [CAR, (0.8, 0.8, 0.9, 0.9, 0.7)]}] * 10  # no far person; a car of its own
    got = score(candidate, reference)
    assert got["person"]["things"] == 20 and got["person"]["found"] == 0.5 and got["person"]["right"] == 1.0
    assert got["person"]["small"] == {"of": 10, "found": 0.0} and got["person"]["large"] == {"of": 10, "found": 1.0}
    assert 0.9 < got["person"]["overlap"] < 1.0
    assert got["vehicle"]["found"] == 1.0 and got["vehicle"]["right"] == 0.5  # half its boxes are of nothing the reference sees


def test_unsure_boxes_do_not_count_as_things_but_excuse_a_candidate():
    reference = [{"person": [NEAR[:4] + (0.3,)], "vehicle": []}]  # the reference is not sure there is anyone
    got = score([{"person": [NEAR], "vehicle": []}], reference)
    assert got["person"]["things"] == 0 and got["person"]["found"] is None and got["person"]["right"] == 1.0
    got = score([{"person": [NEAR[:4] + (0.35,)], "vehicle": []}], [{"person": [NEAR], "vehicle": []}])
    assert got["person"]["found"] == 0.0  # the candidate too unsure to count as finding it


def test_only_one_at_a_time(tmp_path):
    bench = Bench(tmp_path)
    assert bench.candidates() == [] and bench.start(20, 5) and not bench.start(20, 5)
    assert bench.status()["state"] == "collecting"


def test_what_protect_reported_is_a_second_yardstick():
    reference = [{"person": [NEAR], "vehicle": [CAR]}] * 4
    candidate = [{"person": [NEAR], "vehicle": []}] * 4
    protect = [[("person", (0.12, 0.22, 0.32, 0.62)), ("vehicle", CAR[:4])]] * 4  # Protect's boxes, a little off as they are
    got = score(candidate, reference, protect)
    assert got["person"]["protect"] == {"of": 4, "found": 1.0} and got["vehicle"]["protect"] == {"of": 4, "found": 0.0}
    assert score(candidate, reference)["person"]["protect"] == {"of": 0, "found": None}
