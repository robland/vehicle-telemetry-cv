from monitoring_taxi.vision.detection import PersonDetection
from monitoring_taxi.vision.tracker import SimpleTracker
from monitoring_taxi.vision.tracking_utils import calculate_iou


def make_detection(
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
) -> PersonDetection:
    return PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
    )


def test_crossing_people_create_association_ambiguity():
    tracker = SimpleTracker(max_distance=0.15, max_missed_frames=0)

    # Frame 1
    person_a = make_detection(
        0.20, 0.20,
        0.30, 0.50,
    )

    person_b = make_detection(
        0.60, 0.20,
        0.70, 0.50,
    )

    tracks = tracker.update([person_a, person_b])

    assert len(tracks) == 2

    track_a_id = tracks[0].track_id
    track_b_id = tracks[1].track_id

    # Frame 2 : les personnes se rapprochent
    person_a = make_detection(
        0.32, 0.20,
        0.42, 0.50,
    )

    person_b = make_detection(
        0.48, 0.20,
        0.58, 0.50,
    )
    tracks = tracker.update([person_b, person_a])

    assert len(tracks) == 2
    assert tracks[0].track_id == track_a_id
    assert tracks[1].track_id == track_b_id

    # Vérification de l'ambiguïté avec l'IoU
    iou_a_a = calculate_iou(
        person_a.bbox,
        make_detection(
            0.20, 0.20,
            0.30, 0.50,
        ).bbox,
    )

    iou_b_b = calculate_iou(
        person_b.bbox,
        make_detection(
            0.60, 0.20,
            0.70, 0.50,
        ).bbox,
    )

    assert 0.0 <= iou_a_a <= 1.0
    assert 0.0 <= iou_b_b <= 1.0

    # Les centres restent dans la distance maximale.
    # Le tracker actuel doit donc tenter une association
    # uniquement avec la distance.

    assert tracker.tracks_created == 2
    assert {track.track_id for track in tracks} == {
        track_a_id,
        track_b_id,
    }


def bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    center_x = (bbox["x_min"] + bbox["x_max"]) / 2
    center_y = (bbox["y_min"] + bbox["y_max"]) / 2

    return center_x, center_y


def center_distance(
        bbox1: dict[str, float],
        bbox2: dict[str, float],
) -> float:
    center1 = bbox_center(bbox1)
    center2 = bbox_center(bbox2)

    return (
            (center1[0] - center2[0]) ** 2
            + (center1[1] - center2[1]) ** 2
    ) ** 0.5


def test_crossing_people_distance_can_be_ambiguous():
    previous_a = make_detection(
        0.30, 0.20,
        0.45, 0.60,
    )

    previous_b = make_detection(
        0.55, 0.20,
        0.70, 0.60,
    )

    current_a = make_detection(
        0.47, 0.20,
        0.62, 0.60,
    )

    current_b = make_detection(
        0.38, 0.20,
        0.53, 0.60,
    )

    distance_a_a = center_distance(
        previous_a.bbox,
        current_a.bbox,
    )

    distance_a_b = center_distance(
        previous_a.bbox,
        current_b.bbox,
    )

    distance_b_a = center_distance(
        previous_b.bbox,
        current_a.bbox,
    )

    distance_b_b = center_distance(
        previous_b.bbox,
        current_b.bbox,
    )

    print(
        "\n"
        f"A -> A: {distance_a_a:.3f}\n"
        f"A -> B: {distance_a_b:.3f}\n"
        f"B -> A: {distance_b_a:.3f}\n"
        f"B -> B: {distance_b_b:.3f}"
    )
    iou_a_a = calculate_iou(
        previous_a.bbox,
        current_a.bbox,
    )

    iou_a_b = calculate_iou(
        previous_a.bbox,
        current_b.bbox,
    )

    iou_b_a = calculate_iou(
        previous_b.bbox,
        current_a.bbox,
    )

    iou_b_b = calculate_iou(
        previous_b.bbox,
        current_b.bbox,
    )

    print(
        "\n"
        f"A -> A IoU: {iou_a_a:.3f}\n"
        f"A -> B IoU: {iou_a_b:.3f}\n"
        f"B -> A IoU: {iou_b_a:.3f}\n"
        f"B -> B IoU: {iou_b_b:.3f}"
    )
