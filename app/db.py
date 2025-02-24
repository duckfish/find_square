import models
from config import config
from sqlmodel import SQLModel, create_engine

db_url = f"postgresql://{config.DB_USER}:\
{config.DB_PASSWORD}@{config.DB_HOST}:\
{config.DB_PORT}/{config.DB_NAME}"

engine = create_engine(db_url)


def create_db():
    SQLModel.metadata.create_all(engine)
