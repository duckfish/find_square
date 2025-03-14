import logging
from contextlib import asynccontextmanager

from config import config
from db import create_db
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
    create_db()
    logger.info("Application startup complete.")


async def shutdown():
    # await db.close_database_connection()
    logger.info("Application shutdown complete.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    context = {
        "request": request,
        "host": config.UVICORN_SERVER_HOST,
        "port": str(config.UVICORN_SERVER_PORT),
    }

    return templates.TemplateResponse("index.html", context)
