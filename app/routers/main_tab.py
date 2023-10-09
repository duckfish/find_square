import numpy as np
from cv.image_processing import ImageGenerator, SquareDetector
from dependencies import (
    MongoManager,
    get_database,
    get_image_generator,
    get_square_detector,
)
from fastapi import APIRouter, Depends
from models import ImageDataUpdate, ImageParams, UserData

router = APIRouter(tags=["main"])


@router.post("/generate-image")
async def generate_image(
    user_data: UserData,
    image_params: ImageParams,
    image_generator: ImageGenerator = Depends(get_image_generator),
    db: MongoManager = Depends(get_database),
):
    img = image_generator.generate_img(
        image_params.square_size, image_params.lines_numb, image_params.line_thickness
    )
    image_data = {
        "_id": user_data.id,
        "session_id": user_data.session_id,
        "image": img.tobytes(),
    }
    await db.add_line(image_data)

    img_base64 = image_generator.get_img_base64(img)
    return {"img": img_base64}


@router.post("/find-square")
async def test_image(
    user_data: UserData,
    db: MongoManager = Depends(get_database),
    square_detector: SquareDetector = Depends(get_square_detector),
):
    image_data = await db.get_image(user_data.id)
    image = np.frombuffer(image_data.image, dtype=np.uint8).reshape((1000, 1000))

    image_result = square_detector.find_square(image)
    img_base64 = square_detector.get_img_base64(image_result)

    image_data_update = {"_id": user_data.id, "image_result": image_result.tobytes()}
    image_data_update = ImageDataUpdate(**image_data_update)
    await db.update_line(image_data_update)

    return {"img": img_base64}
