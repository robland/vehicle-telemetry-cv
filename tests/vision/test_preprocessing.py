from PIL import Image
from monitoring_taxi.vision.preprocessing import letterbox


def test_letterbox():
    image = Image.open("tests/data/raw/test_image.png") # W=826 , H=1816
    resized_img, transform = letterbox(image) # W=291, H=640
    assert resized_img.size == (640, 640)
    assert transform == {
        "scale": 640/1816,
        "padding_x": 0,
        "padding_y": 174,
        "original_width": 1816,
        "original_height": 826,
        "target_width": 640,
        "target_height": 640,
    }


def test_letterbox_landscape():
    pass


def test_letterbox_portrait():
    pass

