from pydantic_settings import BaseSettings


class Config(BaseSettings):
    RESULT_PATH: str = "data"
    LOG_CONFIG: str = "app/log_config.yaml"
    UVICORN_SERVER_HOST: str
    UVICORN_SERVER_PORT: int
    UVICORN_SERVER_RELOAD: bool

    class Config:
        env_file = ".env"


config = Config()
