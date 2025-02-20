import uvicorn
from config import config

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.UVICORN_SERVER_HOST,
        port=config.UVICORN_SERVER_PORT,
        log_config=config.LOG_CONFIG,
        reload=config.UVICORN_SERVER_RELOAD,
        access_log=False,
    )
