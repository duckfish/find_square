import logging
from contextlib import asynccontextmanager

import uvicorn
from config import config
from db import SQLModel, engine
from dependencies import db
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routers import main_tab

logger = logging.getLogger("find_square")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


app = FastAPI(lifespan=lifespan)

app.include_router(main_tab.router)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


async def startup():
    SQLModel.metadata.create_all(engine)
    logger.info("Application startup complete.")


async def shutdown():
    await db.close_database_connection()
    logger.info("Application shutdown complete.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    context = {
        "request": request,
        "host": config.UVICORN_SERVER_HOST,
        "port": str(config.UVICORN_SERVER_PORT),
    }

    return templates.TemplateResponse("index.html", context)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.UVICORN_SERVER_HOST,
        port=config.UVICORN_SERVER_PORT,
        log_config=config.LOG_CONFIG,
        reload=config.UVICORN_SERVER_RELOAD,
        access_log=False,
    )
