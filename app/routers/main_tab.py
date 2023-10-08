import cachetools
import cv2
import numpy as np
from cv.image_processing import image_generator, square_detector
from dependencies import MongoManager, get_database
from fastapi import APIRouter, Depends
from models import ImageParams, UserData

router = APIRouter(tags=["main"])

img_cache = cachetools.LRUCache(maxsize=100)


@router.post("/generate-image")
async def generate_image(
    user_data: UserData,
    image_params: ImageParams,
    db: MongoManager = Depends(get_database),
):
    img = image_generator.generate_img(
        image_params.square_size, image_params.lines_numb, image_params.line_thickness
    )
    img_cache[user_data.session_id] = img
    image_data = {
        "_id": user_data.id,
        "session_id": user_data.session_id,
        "image": img.tobytes(),
    }
    await db.add_line(image_data)

    img_base64 = image_generator.get_img_base64(img)
    return {"img": img_base64}


@router.post("/find-square")
async def test_image(user_data: UserData, db: MongoManager = Depends(get_database)):
    # img = img_cache[user_data.session_id]
    # img = cv2.imread(
    #     "/home/duckfish/projects/find_square/image_testing/img_kek2.png",
    #     cv2.IMREAD_GRAYSCALE,
    # )
    image_data = await db.get_image(user_data.id)
    image = np.frombuffer(image_data.image, dtype=np.uint8).reshape((1000, 1000))
    img_base64 = square_detector.find_square(image)
    return {"img": img_base64}
