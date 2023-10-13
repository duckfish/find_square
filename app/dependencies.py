import logging

from config import config
from cv.image_processing import (
    ImageGenerator,
    SquareDetector,
    image_generator,
    square_detector,
)
from models import ImageData, ImageDataUpdate
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger("find_square")


class MongoManager:
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.db: AsyncIOMotorDatabase = None

    async def connect_to_database(self, url: str):
        logger.info("Connecting to MongoDB.")
        self.client = AsyncIOMotorClient(url)
        self.db = self.client[config.DB_NAME]
        logger.info("Connected to MongoDB.")

    async def close_database_connection(self):
        logger.info("Closing connection with MongoDB.")
        self.client.close()
        logger.info("Closed connection with MongoDB.")

    async def get_image(self, id: int) -> ImageData:
        image_q = await self.db.images.find_one({"_id": id})
        if image_q:
            return ImageData(**image_q)

    async def update_line(self, image_data_update: ImageDataUpdate):
        await self.db.images.update_one(
            {"_id": image_data_update.id},
            {"$set": image_data_update.model_dump(exclude={"id"})},
        )

    async def add_line(self, image_data: ImageData):
        await self.db.images.insert_one(image_data)


db = MongoManager()


async def get_database() -> MongoManager:
    return db


async def get_square_detector() -> SquareDetector:
    return square_detector


async def get_image_generator() -> ImageGenerator:
    return image_generator
