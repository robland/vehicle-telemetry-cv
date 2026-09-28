import math

from scipy.optimize import linear_sum_assignment

from monitoring_taxi.vision.detection import PersonDetection


def calculate_iou(bbox1: dict[str, float], bbox2: dict[str, float]) -> float:
    x_min = max(bbox1["x_min"], bbox2["x_min"])
    x_max = min(bbox1["x_max"], bbox2["x_max"])

    y_min = max(bbox1["y_min"], bbox2["y_min"])
    y_max = min(bbox1["y_max"], bbox2["y_max"])

    if (0 <= x_min < x_max) and (0 <= y_min < y_max):
        intersection = calculate_surface((x_min, x_max), (y_min, y_max))
        surface_bbox1 = calculate_surface((bbox1["x_min"], bbox1["x_max"]), (bbox1["y_min"], bbox1["y_max"]))
        surface_bbox2 = calculate_surface((bbox2["x_min"], bbox2["x_max"]), (bbox2["y_min"], bbox2["y_max"]))
        return intersection/(surface_bbox1 + surface_bbox2 - intersection)
    return 0.0


def calculate_surface(x: tuple[float, float], y: tuple[float, float]) -> float:
    return abs((x[1]-x[0]) * (y[1]-y[0]))


def calculate_bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    center_x = (bbox["x_min"] + bbox["x_max"])/2
    center_y = (bbox["y_min"] + bbox["y_max"])/2

    return center_x, center_y


def calculate_center_distance(bbox1: dict[str, float], bbox2: dict[str, float]) -> float:
    center_1 = calculate_bbox_center(bbox1)
    center_2 = calculate_bbox_center(bbox2)

    return math.sqrt(
        (center_1[0] - center_2[0]) ** 2 +
        (center_1[1] - center_2[1]) ** 2

    )


def calculate_association_cost(detection_bbox: dict[str, float], track_bbox: dict[str, float], alpha: float = 0.5) -> float:
    if not 0 <= alpha <= 1:
        raise ValueError(
            f"Expected alpha between [0, 1] but got {alpha}."
        )
    center_distance = calculate_center_distance(
        detection_bbox,
        track_bbox
    )

    iou = calculate_iou(
        detection_bbox,
        track_bbox
    )

    return alpha * center_distance + (1 - alpha) * (1 - iou)


def build_cost_matrix(tracks: list, detections: list[PersonDetection], alpha: float = 0.5) -> list[list[float]]:
    return [
        [calculate_association_cost(
            detection_bbox=detection.bbox,
            track_bbox=track.detection.bbox,
            alpha=alpha
        ) for detection in detections
        ] for track in tracks
    ]


def hungarian_assignment(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    if not cost_matrix:
        return []

    if not cost_matrix[0]:
        return []

    row_indices, column_indices = linear_sum_assignment(
        cost_matrix
    )
    return list(zip(row_indices, column_indices))


def filter_matches(matches: list[tuple[int, int]],
                   cost_matrix: list[list[float]],
                   association_threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if association_threshold < 0:
        raise ValueError(
            "Expected association_threshold >= 0."
        )

    accepted_matches = []
    matched_tracks = set()
    matched_detections = set()

    for track_index, detection_index in matches:
        cost = cost_matrix[track_index][detection_index]

        if cost <= association_threshold:
            accepted_matches.append(
                (track_index, detection_index)
            )
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

    all_tracks = set(range(len(cost_matrix)))
    number_of_detections = (
        len(cost_matrix[0]) if cost_matrix else 0
    )
    all_detections = set(range(number_of_detections))
    unmatched_tracks = sorted(
        all_tracks - matched_tracks
    )
    unmatched_detections = sorted(
        all_detections - matched_detections
    )

    return (
        accepted_matches,
        unmatched_tracks,
        unmatched_detections,
    )

