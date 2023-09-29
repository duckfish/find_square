import cachetools
import cv2
from cv.image_processing import image_generator, square_detector
from fastapi import APIRouter, Form

router = APIRouter(tags=["main"])

img_cache = cachetools.LRUCache(maxsize=100)


@router.post("/generate-image")
def generate_image(
    session_id: str,
    square_size: int = Form(),
    lines_number: int = Form(),
    line_width: int = Form(),
):
    img = image_generator.generate_img(square_size, lines_number, line_width)
    img_cache[session_id] = img
    img_base64 = image_generator.get_img_base64(img)
    return {"img": img_base64}


@router.get("/find-square")
def test_image(
    session_id: str,
):
    img = img_cache[session_id]
    img = cv2.imread(
        "/home/duckfish/projects/find_square/test_img/img.png", cv2.IMREAD_GRAYSCALE
    )
    img_base64 = square_detector.find_square(img)
    return {"img": img_base64}
