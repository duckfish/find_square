import logging

from config import config
from models import ImageData
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger("pet")


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

    # async def get_posts(self) -> List[PostDB]:
    #     posts_list = []
    #     posts_q = self.db.posts.find()
    #     async for post in posts_q:
    #         posts_list.append(PostDB(**post, id=post['_id']))
    #     return posts_list

    async def get_image(self, id: int) -> ImageData:
        image_q = await self.db.images.find_one({"_id": id})
        if image_q:
            return ImageData(**image_q)
        # return image_q

    # async def delete_post(self, post_id: OID):
    #     await self.db.posts.delete_one({'_id': ObjectId(post_id)})

    # async def update_post(self, post_id: OID, post: PostDB):
    #     await self.db.posts.update_one({'_id': ObjectId(post_id)},
    #                                    {'$set': post.dict(exclude={'id'})})

    async def add_line(self, image: ImageData):
        await self.db.images.insert_one(image)


db = MongoManager()


async def get_database() -> MongoManager:
    return db
