from monitoring_taxi.vision.detection import PersonDetection
from monitoring_taxi.vision.tracker import SimpleTracker


def test_tracker_creates_track():
    tracker = SimpleTracker()

    detection = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.2,
            "y_min": 0.2,
            "x_max": 0.4,
            "y_max": 0.6,
        },
    )

    tracks = tracker.update([detection])

    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_tracker_keeps_track_id():
    tracker = SimpleTracker()

    detection_1 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.2,
            "y_min": 0.2,
            "x_max": 0.4,
            "y_max": 0.6,
        },
    )

    detection_2 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.21,
            "y_min": 0.2,
            "x_max": 0.41,
            "y_max": 0.6,
        },
    )

    tracks_1 = tracker.update([detection_1])
    tracks_2 = tracker.update([detection_2])

    assert tracks_1[0].track_id == 1
    assert tracks_2[0].track_id == 1


def test_tracker_creates_new_track_for_distant_detection():
    tracker = SimpleTracker(max_distance=0.1)

    detection_1 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.1,
            "y_min": 0.2,
            "x_max": 0.2,
            "y_max": 0.6,
        },
    )

    detection_2 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.7,
            "y_min": 0.2,
            "x_max": 0.8,
            "y_max": 0.6,
        },
    )

    tracks_1 = tracker.update([detection_1])
    tracks_2 = tracker.update([detection_2])

    assert tracks_1[0].track_id == 1
    assert tracks_2[0].track_id == 2


def test_tracker_tracks_multiple_people():
    tracker = SimpleTracker()

    detection_1 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.1,
            "y_min": 0.2,
            "x_max": 0.2,
            "y_max": 0.6,
        },
    )

    detection_2 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.7,
            "y_min": 0.2,
            "x_max": 0.8,
            "y_max": 0.6,
        },
    )

    tracks = tracker.update([detection_1, detection_2])

    assert len(tracks) == 2
    assert tracks[0].track_id == 1
    assert tracks[1].track_id == 2


def test_tracker_counts_created_tracks():
    tracker = SimpleTracker(max_distance=0.1)

    detection = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.1,
            "y_min": 0.1,
            "x_max": 0.2,
            "y_max": 0.2,
        },
    )

    tracker.update([detection])
    assert tracker.tracks_created == 1

    tracker.update([detection])
    assert tracker.tracks_created == 1


def test_tracker_creates_new_id_when_distance_is_too_large():

    tracker = SimpleTracker(max_distance=0.1)

    detection_1 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.2,
            "y_min": 0.2,
            "x_max": 0.4,
            "y_max": 0.6,
        },
    )

    detection_2 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.5,
            "y_min": 0.2,
            "x_max": 0.7,
            "y_max": 0.6,
        },
    )

    tracks_1 = tracker.update([detection_1])
    tracks_2 = tracker.update([detection_2])

    assert tracks_1[0].track_id == 1
    assert tracks_2[0].track_id == 2



def test_tracker_keeps_id_with_small_bbox_variation():

    tracker = SimpleTracker(max_distance=0.1)

    detection_1 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.2,
            "y_min": 0.2,
            "x_max": 0.4,
            "y_max": 0.6,
        },
    )

    detection_2 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.85,
        bbox={
            "x_min": 0.21,
            "y_min": 0.205,
            "x_max": 0.41,
            "y_max": 0.605,
        },
    )

    tracks_1 = tracker.update([detection_1])
    tracks_2 = tracker.update([detection_2])

    assert tracks_1[0].track_id == 1
    assert tracks_2[0].track_id == 1


def test_tracker_keeps_id_through_small_movements():

    tracker = SimpleTracker(max_distance=0.1)

    detections = [
        {
            "x_min": 0.20,
            "y_min": 0.20,
            "x_max": 0.40,
            "y_max": 0.60,
        },
        {
            "x_min": 0.21,
            "y_min": 0.20,
            "x_max": 0.41,
            "y_max": 0.60,
        },
        {
            "x_min": 0.22,
            "y_min": 0.20,
            "x_max": 0.42,
            "y_max": 0.60,
        },
    ]

    track_ids = []

    for bbox in detections:

        detection = PersonDetection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=bbox,
        )

        tracks = tracker.update([detection])

        track_ids.append(tracks[0].track_id)

    assert track_ids == [1, 1, 1]


def test_tracker_keeps_track_when_detection_is_missing():

    tracker = SimpleTracker(
        max_distance=0.1,
        max_missed_frames=1,
    )

    detection = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.2,
            "y_min": 0.2,
            "x_max": 0.4,
            "y_max": 0.6,
        },
    )

    tracks = tracker.update([detection])

    assert tracks[0].track_id == 1


def test_tracker_recovers_same_id_after_missing_detection():

    tracker = SimpleTracker(
        max_distance=0.1,
        max_missed_frames=1,
    )

    detection_1 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.2,
            "y_min": 0.2,
            "x_max": 0.4,
            "y_max": 0.6,
        },
    )

    detection_2 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.21,
            "y_min": 0.2,
            "x_max": 0.41,
            "y_max": 0.6,
        },
    )

    tracks_1 = tracker.update([detection_1])

    tracks_2 = tracker.update([])

    tracks_3 = tracker.update([detection_2])

    assert tracks_1[0].track_id == 1
    assert tracks_2[0].track_id == 1
    assert tracks_3[0].track_id == 1