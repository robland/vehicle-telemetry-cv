def check_bbox(bbox: dict, image_width: int, image_height: int) -> None:
    if image_width <= 0:
        raise ValueError(f"Expected image_width > 0 but got {image_width}")

    if image_height <= 0:
        raise ValueError(f"Expected image_height > 0 but got {image_height}")

    if bbox["x_min"] > image_width:
        raise ValueError(f"Expected x_min <= image_width but got {bbox['x_min']} > {image_width}")

    if bbox["y_min"] > image_height:
        raise ValueError(f"Expected y_min <= image_height but got {bbox['y_min']} > {image_height}")

    if bbox["x_max"] > image_width:
        raise ValueError(f"Expected x_max <= image_width but got {bbox['x_max']} > {image_width}")

    if bbox["y_max"] > image_height:
        raise ValueError(f"Expected y_max <= image_height but got {bbox['y_max']} > {image_height}")

    for k, v in bbox.items():
        if bbox[k] < 0:
            raise ValueError(f"Expected {k} >= 0 but got {v}")

    if bbox["x_min"] >= bbox["x_max"]:
        raise ValueError(f"Expected x_min < x_max but got {bbox['x_min']} >= {bbox['x_max']}")

    if bbox["y_min"] >= bbox["y_max"]:
        raise ValueError(f"Expected y_min < y_max but got {bbox['y_min']} >= {bbox['y_max']}")


def bbox_to_original(bbox: dict, transform: dict) -> dict:
    scale = transform.get("scale")
    original_width = transform.get("original_width")
    original_height = transform.get("original_height")
    padding_x = transform.get("padding_x")
    padding_y = transform.get("padding_y")

    if scale is None or scale <= 0:
        raise ValueError(f"Expected scale but got {scale}")

    results = {
        "x_min": (bbox["x_min"] - padding_x) / scale,
        "x_max": (bbox["x_max"] - padding_x) / scale,
        "y_min": (bbox["y_min"] - padding_y) / scale,
        "y_max": (bbox["y_max"] - padding_y) / scale
    }

    for k, v in results.items():
        vmax = original_width if k.startswith("x") else original_height
        # if v < 0:
        #    results[k] = 0
        # if v > vmax:
        #    results[k] = vmax
        results[k] = max(0, min(v, vmax))
    return results


def normalize_bbox(bbox: dict, image_width: int, image_height: int) -> dict:
    check_bbox(bbox, image_width, image_height)
    return {
        "x_min": bbox["x_min"] / image_width,
        "y_min": bbox["y_min"] / image_height,
        "x_max": bbox["x_max"] / image_width,
        "y_max": bbox["y_max"] / image_height,
    }

