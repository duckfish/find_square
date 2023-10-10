import numpy as np
from cv.image_processing import ImageGenerator, SquareDetector
from dependencies import (
    MongoManager,
    get_database,
    get_image_generator,
    get_square_detector,
)
from fastapi import APIRouter, Depends
from models import (
    ImageCreateRequest,
    ImageDataUpdate,
    ImageFindRequest,
    ImageParams,
    UserData,
)

router = APIRouter(tags=["main"])


@router.post("/generate-image")
async def generate_image(
    image_create: ImageCreateRequest,
    image_generator: ImageGenerator = Depends(get_image_generator),
    db: MongoManager = Depends(get_database),
):
    img = image_generator.generate_img(
        image_create.square_size, image_create.lines_numb, image_create.line_thickness
    )
    image_data = {
        "_id": image_create.id,
        "session_id": image_create.session_id,
        "image": img.tobytes(),
    }
    await db.add_line(image_data)

    img_base64 = image_generator.get_img_base64(img)
    return {"img": img_base64}


@router.post("/find-square")
async def test_image(
    image_find: ImageFindRequest,
    db: MongoManager = Depends(get_database),
    square_detector: SquareDetector = Depends(get_square_detector),
):
    image_data = await db.get_image(image_find.id)
    image = np.frombuffer(image_data.image, dtype=np.uint8).reshape((1000, 1000))

    image_result, elapsed_time = square_detector.find_square(
        image, image_find.ransac_iterations
    )
    img_base64 = square_detector.get_img_base64(image_result)

    image_data_update = {
        "_id": image_find.id,
        "image_result": image_result.tobytes(),
        "elapsed_time": elapsed_time,
    }
    image_data_update = ImageDataUpdate(**image_data_update)
    await db.update_line(image_data_update)

    return {"img": img_base64, "elapsed_time": elapsed_time}
