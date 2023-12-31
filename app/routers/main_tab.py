import numpy as np
from config import config
from cv.image_processing import ImageGenerator, SquareDetector
from dependencies import (
    MongoManager,
    get_database,
    get_image_generator,
    get_square_detector,
)
from fastapi import APIRouter, Depends
from models import ImageCreateRequest, ImageDataUpdate, ImageFindRequest

router = APIRouter(tags=["main"])


@router.post(
    "/generate-image",
    summary="Generate an image",
    description="Generate an image with following params: square size, lines number,\
    line thickness",
)
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
        "size": image_create.square_size,
        "lines": image_create.lines_numb,
        "thickness": image_create.line_thickness,
    }
    await db.add_line(image_data)

    img_base64 = image_generator.get_img_base64(img)
    return {"img": img_base64}


@router.post(
    "/find-square",
    summary="Find square",
)
async def test_image(
    image_find: ImageFindRequest,
    db: MongoManager = Depends(get_database),
    square_detector: SquareDetector = Depends(get_square_detector),
):
    image_data = await db.get_image(image_find.id)
    image = np.frombuffer(image_data.image, dtype=np.uint8).reshape(
        (config.IMG_SIZE, config.IMG_SIZE)
    )

    detector = image_find.detector
    image_result, elapsed_time = square_detector.find_square(
        image, ransac_iterations=image_find.ransac_iterations, detector=detector
    )
    if image_result is not None:
        img_base64 = square_detector.get_img_base64(image_result)
        success = True
    else:
        img_base64 = square_detector.get_img_base64(image)
        success = False

    image_data_update = {
        "_id": image_find.id,
        "ransac_iterations": image_find.ransac_iterations,
        "detector": image_find.detector,
        "success": success,
        "elapsed_time": elapsed_time,
    }
    image_data_update = ImageDataUpdate(**image_data_update)
    await db.update_line(image_data_update)

    return {"img": img_base64, "elapsed_time": elapsed_time, "success": success}
