from uuid import uuid4

import cv2
from config import config
from cv.image_processing import ImageGenerator, SquareDetector
from dependencies import get_image_generator, get_session, get_square_detector
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from models import ImageCreateRequest, ImageFindRequest, SquareDetection
from sqlmodel import Session, select

router = APIRouter(tags=["main"])


@router.post(
    "/generate-image",
    summary="Generate an image",
    description="Generate an image with following params: square size, lines number,\
    line thickness",
)
async def generate_image(
    request: Request,
    response: Response,
    image_create: ImageCreateRequest,
    image_generator: ImageGenerator = Depends(get_image_generator),
    session: Session = Depends(get_session),
):
    user_session = request.cookies.get("user_session")
    if not user_session:
        user_session = str(uuid4())
        response.set_cookie(key="user_session", value=user_session, httponly=True)

    request_id = str(uuid4())
    response.set_cookie(key="request_id", value=request_id, httponly=True)

    db_result = SquareDetection.model_validate(image_create)

    db_result.user_session = user_session
    db_result.request_id = request_id

    img, img_path = image_generator.generate_img(
        image_create.square_size, image_create.lines_qty, image_create.lines_thickness
    )
    db_result.img_path = img_path

    with session:
        session.add(db_result)
        session.commit()
        session.refresh(db_result)

    img_base64 = image_generator.get_img_base64(img)
    return {"img": img_base64}


@router.post(
    "/find-square",
    summary="Find square",
)
async def test_image(
    request: Request,
    image_find: ImageFindRequest,
    session: Session = Depends(get_session),
    square_detector: SquareDetector = Depends(get_square_detector),
):
    request_id = request.cookies.get("request_id")
    if not request_id:
        raise HTTPException(status_code=404, detail="No image found for this session")

    print(request_id)

    with session:
        statement = select(SquareDetection).where(
            SquareDetection.request_id == request_id
        )
        square_detection = session.exec(statement).one()
        img_path = square_detection.img_path

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # type: ignore
    image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))  # type: ignore

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

    square_detection.ransac_iterations = image_find.ransac_iterations
    square_detection.detector = image_find.detector
    square_detection.success = success
    square_detection.elapsed_time = elapsed_time

    with session:
        session.add(square_detection)
        session.commit()
        session.refresh(square_detection)

    return {"img": img_base64, "elapsed_time": elapsed_time, "success": success}
