import pytest

from monitoring_taxi.vision.bbox import bbox_to_original, normalize_bbox


def test_bbox_to_orginal():
    transform = {
        "scale": 1/3,
        "padding_x": 0,
        "padding_y": 140,
        "original_width": 1920,
        "original_height": 1080,
        "target_width": 640,
        "target_height": 640,
    }
    bbox = {
        "x_min": 160,
        "y_min": 170,
        "x_max": 320,
        "y_max": 350
    }

    results = bbox_to_original(bbox, transform)
    assert results == {
        "x_min": 480,
        "x_max": 960,
        "y_min": 90,
        "y_max": 630
    }


def test_bbox_with_extended_padding():
    transform = {
        "scale": 1/3,
        "padding_x": 0,
        "padding_y": 140,
        "original_width": 1920,
        "original_height": 1080,
        "target_width": 640,
        "target_height": 640,
    }
    bbox = {
        "x_min": 100,
        "y_min": 50,
        "x_max": 300,
        "y_max": 200
    }
    results = bbox_to_original(bbox, transform)
    assert results["y_min"] == 0


def test_bbox_that_extends_image():
    transform = {
        "scale": 1/3,
        "padding_x": 0,
        "padding_y": 140,
        "original_width": 1920,
        "original_height": 1080,
        "target_width": 640,
        "target_height": 640,
    }
    bbox = {
        "x_min": 1800,
        "y_min": 100,
        "x_max": 2000,
        "y_max": 1200
    }
    results = bbox_to_original(bbox, transform)
    assert results["x_max"] == 1920
    assert results["y_max"] == 1080


def test_bbox_entirely_within_padding():
    transform = {
        "scale": 1/3,
        "padding_x": 0,
        "padding_y": 140,
        "original_width": 1920,
        "original_height": 1080,
        "target_width": 640,
        "target_height": 640,
    }
    bbox = {
        "x_min": 100,
        "y_min": 20,
        "x_max": 300,
        "y_max": 100
    }
    results = bbox_to_original(bbox, transform)
    assert results["y_min"] == 0
    assert results["y_max"] == 0


def test_normalize_bbox_nominal_case():
    # 480, 90, 960, 630
    # 1920 × 1080
    bbox = {
        "x_min": 480,
        "y_min": 90,
        "x_max": 960,
        "y_max": 630
    }
    image_width = 1920
    image_height = 1080
    normalized_bbox = normalize_bbox(bbox, image_width, image_height)
    assert normalized_bbox == pytest.approx(
        {
            "x_min": 0.25,
            "y_min": 0.083333,
            "x_max": 0.5,
            "y_max": 0.583333
        },
        rel=0.001
    )


def test_normalize_bbox_at_edge_image():
    bbox = {
        "x_min": 0,
        "y_min": 0,
        "x_max": 1920,
        "y_max": 1080
    }
    image_width = 1920
    image_height = 1080
    normalized_bbox = normalize_bbox(bbox, image_width, image_height)
    assert normalized_bbox == {
        "x_min": 0,
        "y_min": 0,
        "x_max": 1,
        "y_max": 1
    }


def test_normalize_bbox_x_min_greater_than_x_max():
    bbox = {
        "x_min": 1920,
        "y_min": 108,
        "x_max": 192,
        "y_max": 108
    }
    image_width = 1920
    image_height = 1080

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, image_width, image_height)
    assert str(e.value) == f"Expected x_min < x_max but got {bbox['x_min']} >= {bbox['x_max']}"


def test_normalize_bbox_x_min_greater_than_image_width():
    bbox = {
        "x_min": 2000,
        "y_min": 480,
        "x_max": 192,
        "y_max": 680
    }
    image_width = 1920
    image_height = 1080

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, image_width, image_height)
    assert str(e.value) == f"Expected x_min <= image_width but got {bbox['x_min']} > {image_width}"


def test_normalize_bbox_x_max_greater_than_image_width():
    bbox = {
        "x_min": 1920,
        "y_min": 108,
        "x_max": 2000,
        "y_max": 108
    }
    image_width = 1920
    image_height = 1080

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, image_width, image_height)
    assert str(e.value) == f"Expected x_max <= image_width but got {bbox['x_max']} > {image_width}"


def test_normalize_bbox_y_min_greater_than_y_max():
    bbox = {
        "x_min": 192,
        "y_min": 1080,
        "x_max": 1920,
        "y_max": 108
    }
    image_width = 1920
    image_height = 1080

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, image_width, image_height)
    assert str(e.value) == f"Expected y_min < y_max but got {bbox['y_min']} >= {bbox['y_max']}"


def test_normalize_bbox_y_min_greater_than_image_height():
    bbox = {
        "x_min": 192,
        "y_min": 2000,
        "x_max": 1920,
        "y_max": 108
    }
    image_width = 1920
    image_height = 1080

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, image_width, image_height)
    assert str(e.value) == f"Expected y_min <= image_height but got {bbox['y_min']} > {image_height}"


def test_normalize_bbox_y_max_greater_than_image_height():
    bbox = {
        "x_min": 192,
        "y_min": 1080,
        "x_max": 1920,
        "y_max": 1920
    }
    image_width = 1920
    image_height = 1080

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, image_width, image_height)
    assert str(e.value) == f"Expected y_max <= image_height but got {bbox['y_max']} > {image_height}"


def test_normalize_bbox_with_null_image_height():
    bbox = {
        "x_min": 1920,
        "y_min": 108,
        "x_max": 192,
        "y_max": 108
    }
    image_width = 1920
    image_height = 0

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, image_width, image_height)
    assert str(e.value) == f"Expected image_height > 0 but got {image_height}"


def test_normalize_bbox_with_null_image_width():
    bbox = {
        "x_min": 1920,
        "y_min": 108,
        "x_max": 192,
        "y_max": 108
    }
    image_width = 0
    image_height = 1080

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, image_width, image_height)
    assert str(e.value) == f"Expected image_width > 0 but got {image_width}"


def test_normalize_bbox_x_min_equal_x_max():
    bbox = {
        "x_min": 500,
        "y_min": 100,
        "x_max": 500,
        "y_max": 600
    }

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, 1920, 1080)

    assert str(e.value) == (
        f"Expected x_min < x_max but got "
        f"{bbox['x_min']} >= {bbox['x_max']}"
    )


def test_normalize_bbox_y_min_equal_y_max():
    bbox = {
        "x_min": 50,
        "y_min": 100,
        "x_max": 500,
        "y_max": 100
    }

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, 1920, 1080)

    assert str(e.value) == (
        f"Expected y_min < y_max but got "
        f"{bbox['y_min']} >= {bbox['y_max']}"
    )


def test_normalize_bbox_negative_x_min():
    bbox = {
        "x_min": -1,
        "y_min": 100,
        "x_max": 500,
        "y_max": 600
    }

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, 1920, 1080)

    assert str(e.value) == (
        f"Expected x_min >= 0 but got {bbox['x_min']}"
    )


def test_normalize_bbox_negative_y_min():
    bbox = {
        "x_min": 200,
        "y_min": -100,
        "x_max": 500,
        "y_max": 600
    }

    with pytest.raises(ValueError) as e:
        normalize_bbox(bbox, 1920, 1080)

    assert str(e.value) == (
        f"Expected y_min >= 0 but got {bbox['y_min']}"
    )