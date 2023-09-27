from config import config
from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.post("/generate-image")
def generate_image(square_size: int, lines_number: int, line_width: int):
    print(square_size, lines_number, line_width)
    # img_base64_roi = tools.bytes_to_base64(img_bytes_roi)
    return {}
