from pydantic_settings import BaseSettings


class Config(BaseSettings):
    UVICORN_SERVER_HOST: str
    UVICORN_SERVER_PORT: int
    UVICORN_SERVER_RELOAD: bool

    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    LOG_CONFIG: str = "app/log_config.yaml"
    IMG_SIZE: int
    MODEL_PATH: str

    class Config:
        env_file = "find_square.env"


config = Config()  # type: ignore
