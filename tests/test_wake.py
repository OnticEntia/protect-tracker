import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("wake", Path(__file__).resolve().parents[1] / "custom_components/protect_tracker/wake.py")
wake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wake)

PARKED = {"tracks": [{"id": "1", "objectType": "vehicle", "stationary": True}]}
WALKING = {"tracks": [{"id": "2", "objectType": "person", "stationary": False}, {"id": "1", "stationary": True}]}


def test_parked_things_wake_nothing():
    w = wake.Waker()
    assert not w.report("cam", PARKED, 0) and not w.awake
    assert not w.report("cam", {"status": "connected"}, 0)
    assert not w.report("cam", {"tracks": []}, 0)


def test_wakes_once_and_sleeps_after_the_idle_time():
    w = wake.Waker(idle=30)
    assert w.report("cam", WALKING, 100)
    assert not w.report("cam", WALKING, 101)  # already awake
    assert w.sweep(120) == []
    w.report("cam", PARKED, 125)  # still there, but nothing moves
    assert w.sweep(131) == [] and w.sweep(131.5) == ["cam"]
    assert w.sweep(200) == []
    assert w.report("cam", WALKING, 300)  # and wakes again


def test_cameras_are_separate():
    w = wake.Waker(idle=10)
    w.report("a", WALKING, 0)
    w.report("b", WALKING, 8)
    assert w.sweep(12) == ["a"] and list(w.awake) == ["b"]
