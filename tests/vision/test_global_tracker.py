from monitoring_taxi.vision.tracker import GlobalTracker
from vision.test_tracking_association import make_detection


def test_global_tracker_creates_tracks():
    tracker = GlobalTracker(
        alpha=0.5,
        association_threshold=0.5,
    )

    detections = [
        make_detection(0.1, 0.1, 0.2, 0.2),
        make_detection(0.7, 0.7, 0.8, 0.8),
    ]

    tracks = tracker.update(detections)

    assert len(tracks) == 2
    assert [track.track_id for track in tracks] == [1, 2]
    assert tracker.tracks_created == 2


def test_global_tracker_keeps_track_id():
    tracker = GlobalTracker(
        alpha=0.5,
        association_threshold=0.5,
    )

    detections = [
        make_detection(0.1, 0.1, 0.2, 0.2),
    ]

    tracks = tracker.update(detections)

    assert tracks[0].track_id == 1

    detections = [
        make_detection(0.11, 0.1, 0.21, 0.2),
    ]

    tracks = tracker.update(detections)

    assert tracks[0].track_id == 1


def test_global_tracker_keeps_multiple_ids():
    tracker = GlobalTracker(
        alpha=0.5,
        association_threshold=0.5,
    )

    tracks = tracker.update([
        make_detection(0.1, 0.1, 0.2, 0.2),
        make_detection(0.7, 0.7, 0.8, 0.8),
    ])

    assert [track.track_id for track in tracks] == [1, 2]

    tracks = tracker.update([
        make_detection(0.12, 0.1, 0.22, 0.2),
        make_detection(0.68, 0.7, 0.78, 0.8),
    ])

    assert [track.track_id for track in tracks] == [1, 2]


def test_global_tracker_creates_new_track():
    tracker = GlobalTracker(
        alpha=0.5,
        association_threshold=0.5,
    )

    tracker.update([
        make_detection(0.1, 0.1, 0.2, 0.2),
    ])

    tracks = tracker.update([
        make_detection(0.11, 0.1, 0.21, 0.2),
        make_detection(0.7, 0.7, 0.8, 0.8),
    ])

    assert [track.track_id for track in tracks] == [1, 2]
    assert tracker.tracks_created == 2


def test_global_tracker_keeps_missed_track():
    tracker = GlobalTracker(
        max_missed_frames=1,
        alpha=0.5,
        association_threshold=0.5,
    )

    tracks = tracker.update([
        make_detection(0.1, 0.1, 0.2, 0.2),
    ])

    assert tracks[0].track_id == 1

    tracks = tracker.update([])

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].frames_since_update == 1


def test_global_tracker_removes_track_after_max_missed_frames():
    tracker = GlobalTracker(
        max_missed_frames=1,
        alpha=0.5,
        association_threshold=0.5,
    )

    tracker.update([
        make_detection(0.1, 0.1, 0.2, 0.2),
    ])

    tracker.update([])

    tracks = tracker.update([])

    assert tracks == []


def test_global_tracker_uses_global_assignment():
    tracker = GlobalTracker(
        alpha=0.5,
        association_threshold=0.5,
    )

    tracks = tracker.update([
        make_detection(0.10, 0.10, 0.20, 0.20),
        make_detection(0.40, 0.10, 0.50, 0.20),
    ])

    assert [track.track_id for track in tracks] == [1, 2]

    tracks = tracker.update([
        make_detection(0.39, 0.10, 0.49, 0.20),
        make_detection(0.11, 0.10, 0.21, 0.20),
    ])

    tracks_by_x = sorted(
        tracks,
        key=lambda track: track.detection.bbox["x_min"],
    )

    assert tracks_by_x[0].track_id == 1
    assert tracks_by_x[1].track_id == 2


def test_global_tracker_handles_detection_order_change():
    tracker = GlobalTracker(
        alpha=0.5,
        association_threshold=0.5,
    )

    person_a = make_detection(
        0.10, 0.10,
        0.20, 0.20,
    )

    person_b = make_detection(
        0.40, 0.10,
        0.50, 0.20,
    )

    tracks = tracker.update([
        person_a,
        person_b,
    ])

    assert tracks[0].track_id == 1
    assert tracks[1].track_id == 2

    # Même personnes, mais ordre des détections inversé.
    person_a_moved = make_detection(
        0.11, 0.10,
        0.21, 0.20,
    )

    person_b_moved = make_detection(
        0.39, 0.10,
        0.49, 0.20,
    )

    tracks = tracker.update([
        person_b_moved,
        person_a_moved,
    ])

    tracks_by_position = sorted(
        tracks,
        key=lambda track: track.detection.bbox["x_min"],
    )

    assert tracks_by_position[0].track_id == 1
    assert tracks_by_position[1].track_id == 2


def test_global_tracker_handles_crossing():
    tracker = GlobalTracker(
        alpha=0.5,
        association_threshold=0.5,
    )

    person_a = make_detection(
        0.10, 0.10,
        0.20, 0.30,
    )

    person_b = make_detection(
        0.30, 0.10,
        0.40, 0.30,
    )

    tracks = tracker.update([
        person_a,
        person_b,
    ])

    assert tracks[0].track_id == 1
    assert tracks[1].track_id == 2

    # Les personnes se rapprochent.
    person_a = make_detection(
        0.17, 0.10,
        0.27, 0.30,
    )

    person_b = make_detection(
        0.23, 0.10,
        0.33, 0.30,
    )

    tracks = tracker.update([
        person_a,
        person_b,
    ])

    # On identifie les tracks par leur position.
    tracks_by_position = sorted(
        tracks,
        key=lambda track: track.detection.bbox["x_min"],
    )

    assert tracks_by_position[0].track_id == 1
    assert tracks_by_position[1].track_id == 2


