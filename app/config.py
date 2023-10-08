from pydantic_settings import BaseSettings


class Config(BaseSettings):
    UVICORN_SERVER_HOST: str
    UVICORN_SERVER_PORT: int
    UVICORN_SERVER_RELOAD: bool
    DB_URL: str
    DB_NAME: str
    LOG_CONFIG: str = "app/log_config.yaml"
    class Config:
        env_file = ".env"


config = Config()
