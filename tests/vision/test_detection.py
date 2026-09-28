import pytest

from monitoring_taxi.vision.detection import PersonDetection


def test_person_detection_valid():
    detection = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.91,
        bbox={
            "x_min": 0.2,
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        },
    )

    assert detection.class_id == 0
    assert detection.class_name == "person"
    assert detection.confidence == 0.91
    assert detection.bbox == {
        "x_min": 0.2,
        "y_min": 0.1,
        "x_max": 0.5,
        "y_max": 0.8,
    }


def test_person_detection_invalid_class_id_negative():
    data = {
        "class_id": -1,
        "class_name": "person",
        "confidence": 0.91,
        "bbox": {
            "x_min": 0.2,
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }
    with pytest.raises(ValueError) as e:
        PersonDetection(**data)

    assert str(e.value) == "Expected class_id >= 0 but got -1."


def test_person_detection_invalid_class_id_type():
    data = {
        "class_id": "AB",
        "class_name": "person",
        "confidence": 0.91,
        "bbox": {
            "x_min": 0.2,
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }
    with pytest.raises(TypeError) as e:
        PersonDetection(**data)
    assert str(e.value) == "Expected type int for class_id."


@pytest.mark.parametrize(
    "class_name", [{}, 1, [2, 4]]
)
def test_person_detection_invalid_class_name_type(class_name):
    data = {
        "class_id": 1,
        "class_name": class_name,
        "confidence": 0.91,
        "bbox": {
            "x_min": 0.2,
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }
    with pytest.raises(TypeError) as e:
        PersonDetection(**data)
    assert str(e.value) == "Expected type str for class_name."


def test_person_detection_invalid_class_name_value():
    data = {
        "class_id": 1,
        "class_name": "box",
        "confidence": 0.91,
        "bbox": {
            "x_min": 0.2,
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }
    with pytest.raises(ValueError) as e:
        PersonDetection(**data)
    assert str(e.value) == f"Expected value person but got {data['class_name']}."


def test_person_detection_confidence_below_zero():
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": -1.0,
        "bbox": {
            "x_min": 0.2,
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }
    with pytest.raises(ValueError) as e:
        PersonDetection(**data)
    assert str(e.value) == f"Expected confidence between [0, 1] but got {data['confidence']}."


def test_person_detection_confidence_above_one():
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 1.5,
        "bbox": {
            "x_min": 0.2,
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }
    with pytest.raises(ValueError) as e:
        PersonDetection(**data)
    assert str(e.value) == f"Expected confidence between [0, 1] but got {data['confidence']}."


def test_person_detection_bbox_missing_coordinate_x_min():
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 1.0,
        "bbox": {
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }
    with pytest.raises(ValueError) as e:
        PersonDetection(**data)
    assert str(e.value) == "Expected x_min value."


def test_person_detection_bbox_missing_coordinate_x_max():
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "bbox": {
            "y_min": 0.1,
            "x_min": 0.5,
            "y_max": 0.8,
        }
    }
    with pytest.raises(ValueError) as e:
        PersonDetection(**data)
    assert str(e.value) == "Expected x_max value."


def test_person_detection_bbox_missing_coordinate_y_min():
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "bbox": {
            "x_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }
    with pytest.raises(ValueError) as e:
        PersonDetection(**data)
    assert str(e.value) == "Expected y_min value."


def test_person_detection_bbox_missing_coordinate_y_max():
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "bbox": {
            "y_min": 0.1,
            "x_min": 0.5,
            "x_max": 0.8,
        }
    }
    with pytest.raises(ValueError) as e:
        PersonDetection(**data)
    assert str(e.value) == "Expected y_max value."


def test_person_detection_bbox_x_out_of_range():
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "bbox": {
            "x_min": 1.5,
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 4,
        }
    }

    with pytest.raises(ValueError) as e:
        PersonDetection(**data)
    assert str(e.value) == f"Expected value with constraint 0 <= x_min < x_max <= 1."


def test_person_detection_bbox_y_out_of_range():
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "bbox": {
            "x_min": 0.5,
            "y_min": 0.1,
            "x_max": 0.6,
            "y_max": 4,
        }
    }

    with pytest.raises(ValueError) as e:
        PersonDetection(**data)
    assert str(e.value) == "Expected value with constraint 0 <= y_min < y_max <= 1."


@pytest.mark.parametrize(
    "x_min, x_max",
    [
        (0.8, 0.5),  # x_min > x_max
        (0.5, 0.5),  # x_min == x_max
    ]
)
def test_person_detection_bbox_invalid_x_order(x_min, x_max):
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.91,
        "bbox": {
            "x_min": x_min,
            "y_min": 0.1,
            "x_max": x_max,
            "y_max": 0.8,
        }
    }

    with pytest.raises(ValueError) as e:
        PersonDetection(**data)

    assert str(e.value) == (
        "Expected value with constraint "
        "0 <= x_min < x_max <= 1."
    )


@pytest.mark.parametrize(
    "y_min, y_max",
    [
        (0.8, 0.5),  # y_min > y_max
        (0.5, 0.5),  # y_min == y_max
    ]
)
def test_person_detection_bbox_invalid_y_order(y_min, y_max):
    data = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.91,
        "bbox": {
            "x_min": 0.3,
            "y_min": y_min,
            "x_max": 0.8,
            "y_max": y_max,
        }
    }

    with pytest.raises(ValueError) as e:
        PersonDetection(**data)

    assert str(e.value) == (
        "Expected value with constraint "
        "0 <= y_min < y_max <= 1."
    )


@pytest.mark.parametrize(
    "x_min",
    [-0.1, -1.0],
)
def test_person_detection_bbox_negative_x(x_min):
    data = {
        "class_id": 1,
        "class_name": "person",
        "confidence": 0.91,
        "bbox": {
            "x_min": x_min,
            "y_min": 0.1,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }

    with pytest.raises(ValueError) as e:
        PersonDetection(**data)

    assert str(e.value) == (
        "Expected value with constraint "
        "0 <= x_min < x_max <= 1."
    )


@pytest.mark.parametrize(
    "y_min",
    [-0.1, -1.0],
)
def test_person_detection_bbox_negative_y(y_min):
    data = {
        "class_id": 1,
        "class_name": "person",
        "confidence": 0.91,
        "bbox": {
            "x_min": 0.1,
            "y_min": y_min,
            "x_max": 0.5,
            "y_max": 0.8,
        }
    }

    with pytest.raises(ValueError) as e:
        PersonDetection(**data)

    assert str(e.value) == (
        "Expected value with constraint "
        "0 <= y_min < y_max <= 1."
    )