"""Which cameras there are: taken up, offline and back, removed for good."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("roster", Path(__file__).resolve().parents[1] / "custom_components/protect_tracker/roster.py")
roster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roster)

READY, OFFLINE, NO_STREAM = roster.READY, roster.OFFLINE, roster.NO_STREAM


def test_taken_up_and_told_once():
    r = roster.Roster()
    assert r.update({"a": READY, "b": OFFLINE}) == [("a", None, READY), ("b", None, OFFLINE)]
    assert r.update({"a": READY, "b": OFFLINE}) == []  # nothing new: nothing said
    assert r.ready() == ["a"]


def test_offline_and_back():
    r = roster.Roster()
    r.update({"a": READY})
    assert r.update({"a": OFFLINE}) == [("a", READY, OFFLINE)]
    assert r.ready() == []
    assert r.update({"a": READY}) == [("a", OFFLINE, READY)]


def test_no_stream_is_its_own_state():
    r = roster.Roster()
    r.update({"a": READY})
    assert r.update({"a": NO_STREAM}) == [("a", READY, NO_STREAM)]
    assert "a" in r.cameras and r.ready() == []


def test_removed_only_once_it_has_stayed_away():
    r = roster.Roster(gone_checks=3)
    r.update({"a": READY, "b": READY})
    assert r.update({"b": READY}) == [] and r.update({"b": READY}) == []  # missing twice: still held
    assert "a" in r.cameras
    assert r.update({"b": READY}) == [("a", READY, None)]  # the third time: removed
    assert "a" not in r.cameras
    assert r.update({"b": READY}) == []  # and said once


def test_missing_for_a_moment_is_not_removal():
    r = roster.Roster(gone_checks=3)
    r.update({"a": READY})
    r.update({})
    r.update({})
    assert r.update({"a": READY}) == []  # back before it counted: nothing happened
    r.update({})
    r.update({})
    assert "a" in r.cameras  # (the count started over)


def test_adopted_again_after_removal():
    r = roster.Roster(gone_checks=1)
    r.update({"a": READY})
    assert r.update({}) == [("a", READY, None)]
    assert r.update({"a": READY}) == [("a", None, READY)]


def test_protect_not_to_be_asked_concludes_nothing_at_first():
    r = roster.Roster(gone_checks=2, unknown_checks=3)
    r.update({"a": READY, "b": OFFLINE})
    assert r.update(None) == [] and r.update(None) == []
    assert r.cameras == {"a": READY, "b": OFFLINE}
    assert r.update(None) == [("a", READY, OFFLINE)]  # for a while now: nothing is watched on trust
    assert r.update(None) == []
    assert set(r.cameras) == {"a", "b"}  # but nothing is ever removed on that account
    assert r.update({"a": READY, "b": OFFLINE}) == [("a", OFFLINE, READY)]
