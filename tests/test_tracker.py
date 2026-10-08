from src.detection.tracker import Tracker

from .helpers import make_person


def test_ids_are_stable_while_moving():
    tracker = Tracker()
    ids = None
    for step in range(20):
        persons = [make_person(400 + 5 * step, 300, 400), make_person(1400 - 5 * step, 300, 400)]
        tracker.update(persons)
        current = [p.track_id for p in persons]
        assert ids is None or current == ids
        ids = current
    assert len(set(ids)) == 2


def test_track_survives_short_gap_then_expires():
    tracker = Tracker(max_age=3)
    first = tracker.update([make_person(800, 300, 400)])[0].track_id
    tracker.update([])
    assert tracker.update([make_person(805, 300, 400)])[0].track_id == first
    for _ in range(4):
        tracker.update([])
    assert tracker.update([make_person(805, 300, 400)])[0].track_id != first
