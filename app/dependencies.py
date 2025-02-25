from cv.image_processing import (
    ImageGenerator,
    SquareDetector,
    image_generator,
    square_detector,
)
from db import engine
from sqlmodel import Session


def get_session():
    with Session(engine) as session:
        yield session


async def get_square_detector() -> SquareDetector:
    return square_detector


async def get_image_generator() -> ImageGenerator:
    return image_generator
