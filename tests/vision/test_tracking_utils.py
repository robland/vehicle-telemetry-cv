import pytest

from monitoring_taxi.vision.tracking_utils import (calculate_iou,
                                                   calculate_bbox_center,
                                                   calculate_center_distance,
                                                   calculate_association_cost,
                                                   build_cost_matrix,
                                                   hungarian_assignment,
                                                   filter_matches)

from vision.test_tracking_association import make_detection
from monitoring_taxi.vision.tracker import Track


def test_iou_identical_boxes():
    bbox_a = {
        "x_min": 0.2,
        "y_min": 0.2,
        "x_max": 0.4,
        "y_max": 0.6,
    }
    bbox_b = {
        "x_min": 0.2,
        "y_min": 0.2,
        "x_max": 0.4,
        "y_max": 0.6,
    }

    assert calculate_iou(bbox_a, bbox_b) == 1.0


def test_iou_no_overlap():

    bbox_a = {
        "x_min": 0.1,
        "y_min": 0.1,
        "x_max": 0.2,
        "y_max": 0.2,
    }

    bbox_b = {
        "x_min": 0.8,
        "y_min": 0.8,
        "x_max": 0.9,
        "y_max": 0.9,
    }

    assert calculate_iou(bbox_a, bbox_b) == 0.0


def test_iou_partial_overlap():

    bbox_a = {
        "x_min": 0.0,
        "y_min": 0.0,
        "x_max": 0.5,
        "y_max": 0.5,
    }

    bbox_b = {
        "x_min": 0.25,
        "y_min": 0.25,
        "x_max": 0.75,
        "y_max": 0.75,
    }

    assert calculate_iou(bbox_a, bbox_b) == 1 / 7


def test_calculate_bbox_center():
    bbox = {
        "x_min": 0.2,
        "y_min": 0.3,
        "x_max": 0.6,
        "y_max": 0.7,
    }

    center = calculate_bbox_center(bbox)

    assert center == (0.4, 0.5)


def test_calculate_center_distance_same_bbox():
    bbox = {
        "x_min": 0.2,
        "y_min": 0.3,
        "x_max": 0.6,
        "y_max": 0.7,
    }

    distance = calculate_center_distance(
        bbox,
        bbox,
    )

    assert distance == 0.0


def test_calculate_association_cost_same_bbox():
    bbox = {
        "x_min": 0.2,
        "y_min": 0.3,
        "x_max": 0.6,
        "y_max": 0.7,
    }

    cost = calculate_association_cost(
        bbox,
        bbox,
    )

    assert cost == 0.0


def test_calculate_association_cost_rejects_invalid_alpha():
    bbox = {
        "x_min": 0.2,
        "y_min": 0.3,
        "x_max": 0.6,
        "y_max": 0.7,
    }

    with pytest.raises(ValueError):
        calculate_association_cost(
            bbox,
            bbox,
            alpha=1.5,
        )


def test_build_cost_matrix():
    track_1 = Track(
        track_id=1,
        detection=make_detection(
            x_min=0.1,
            y_min=0.1,
            x_max=0.2,
            y_max=0.2,
        ),
    )

    track_2 = Track(
        track_id=2,
        detection=make_detection(
            x_min=0.7,
            y_min=0.7,
            x_max=0.8,
            y_max=0.8,
        ),
    )

    detection_1 = make_detection(
        x_min=0.1,
        y_min=0.1,
        x_max=0.2,
        y_max=0.2,
    )

    detection_2 = make_detection(
        x_min=0.7,
        y_min=0.7,
        x_max=0.8,
        y_max=0.8,
    )

    matrix = build_cost_matrix(
        tracks=[track_1, track_2],
        detections=[detection_1, detection_2],
    )

    assert len(matrix) == 2
    assert len(matrix[0]) == 2
    assert len(matrix[1]) == 2

    assert matrix[0][0] == 0.0
    assert matrix[1][1] == 0.0

    assert matrix[0][1] > matrix[0][0]
    assert matrix[1][0] > matrix[1][1]


def test_build_cost_matrix_different_number_of_tracks_and_detections():
    track_1 = Track(
        track_id=1,
        detection=make_detection(
            x_min=0.1,
            y_min=0.1,
            x_max=0.2,
            y_max=0.2,
        ),
    )

    track_2 = Track(
        track_id=2,
        detection=make_detection(
            x_min=0.4,
            y_min=0.4,
            x_max=0.5,
            y_max=0.5,
        ),
    )

    detection_1 = make_detection(
        x_min=0.1,
        y_min=0.1,
        x_max=0.2,
        y_max=0.2,
    )

    detection_2 = make_detection(
        x_min=0.4,
        y_min=0.4,
        x_max=0.5,
        y_max=0.5,
    )

    detection_3 = make_detection(
        x_min=0.8,
        y_min=0.8,
        x_max=0.9,
        y_max=0.9,
    )

    matrix = build_cost_matrix(
        tracks=[track_1, track_2],
        detections=[
            detection_1,
            detection_2,
            detection_3,
        ],
    )

    assert len(matrix) == 2
    assert len(matrix[0]) == 3
    assert len(matrix[1]) == 3


def test_hungarian_assignment():
    cost_matrix = [
        [0.10, 0.72, 0.45],
        [0.61, 0.08, 0.53],
        [0.49, 0.57, 0.06],
    ]

    matches = hungarian_assignment(cost_matrix)

    assert matches == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]


def test_hungarian_assignment_is_global():
    cost_matrix = [
        [0.10, 0.20],
        [0.11, 0.90],
    ]

    matches = hungarian_assignment(cost_matrix)

    assert matches == [
        (0, 1),
        (1, 0),
    ]


def test_hungarian_assignment_more_detections_than_tracks():
    cost_matrix = [
        [0.10, 0.80, 0.90],
        [0.70, 0.20, 0.60],
    ]

    matches = hungarian_assignment(cost_matrix)

    assert matches == [
        (0, 0),
        (1, 1),
    ]


def test_hungarian_assignment_more_tracks_than_detections():
    cost_matrix = [
        [0.10, 0.80],
        [0.70, 0.20],
        [0.90, 0.60],
    ]

    matches = hungarian_assignment(cost_matrix)

    assert matches == [
        (0, 0),
        (1, 1),
    ]


def test_hungarian_assignment_empty_matrix():
    assert hungarian_assignment([]) == []


def test_filter_matches():
    cost_matrix = [
        [0.10, 0.80],
        [0.70, 0.20],
    ]

    matches = [
        (0, 0),
        (1, 1),
    ]

    accepted, unmatched_tracks, unmatched_detections = (
        filter_matches(
            matches,
            cost_matrix,
            association_threshold=0.5,
        )
    )

    assert accepted == [
        (0, 0),
        (1, 1),
    ]

    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_filter_matches_rejects_high_cost_matches():
    cost_matrix = [
        [0.10, 0.80],
        [0.70, 0.90],
    ]

    matches = [
        (0, 0),
        (1, 1),
    ]

    accepted, unmatched_tracks, unmatched_detections = (
        filter_matches(
            matches,
            cost_matrix,
            association_threshold=0.5,
        )
    )

    assert accepted == [
        (0, 0),
    ]

    assert unmatched_tracks == [1]
    assert unmatched_detections == [1]


def test_filter_matches_with_new_detection():
    cost_matrix = [
        [0.10, 0.80, 0.90],
        [0.70, 0.20, 0.80],
    ]

    matches = [
        (0, 0),
        (1, 1),
    ]

    accepted, unmatched_tracks, unmatched_detections = (
        filter_matches(
            matches,
            cost_matrix,
            association_threshold=0.5,
        )
    )

    assert accepted == [
        (0, 0),
        (1, 1),
    ]

    assert unmatched_tracks == []
    assert unmatched_detections == [2]


def test_filter_matches_with_missing_detection():
    cost_matrix = [
        [0.10, 0.80],
        [0.70, 0.20],
        [0.90, 0.80],
    ]

    matches = [
        (0, 0),
        (1, 1),
    ]

    accepted, unmatched_tracks, unmatched_detections = (
        filter_matches(
            matches,
            cost_matrix,
            association_threshold=0.5,
        )
    )

    assert accepted == [
        (0, 0),
        (1, 1),
    ]

    assert unmatched_tracks == [2]
    assert unmatched_detections == []

