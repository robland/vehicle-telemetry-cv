from pathlib import Path
from PIL import Image


def letterbox(image: Image, target_width: int = 640, target_height: int = 640) -> Image:
    w, h = image.size
    scale = min(target_height/w, target_width/h)
    resized_w, resized_h = int(w * scale), int(h * scale)

    resized_img = image.resize((resized_w, resized_h))

    padded_img = Image.new("RGB", (target_width, target_height), 0)

    padding_x = int(target_width - resized_w)//2
    padding_y = int(target_height - resized_h)//2

    padded_img.paste(resized_img, (padding_x, padding_y))

    transform = {
        "scale": scale,
        "padding_x": padding_x,
        "padding_y": padding_y,
        "original_width": w,
        "original_height": h,
        "target_width": target_width,
        "target_height": target_height
    }
    return padded_img, transform


if __name__ == "__main__":
    filepath = Path("data/raw/test_image.png")
    image = Image.open(filepath)
    resized_img = letterbox(image)