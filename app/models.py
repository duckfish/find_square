from datetime import datetime

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLModelField
from sqlmodel import SQLModel


class ImageCreateRequest(BaseModel):
    timestamp: datetime
    session_id: str
    square_size: int
    lines_qty: int
    lines_thickness: int


class SquareDetection(SQLModel, table=True):
    request_id: str | None = SQLModelField(default=None, primary_key=True)
    user_session: str | None = None
    # request_id: str | None = None
    timestamp: datetime

    img_path: str | None = None
    square_size: int
    lines_qty: int
    lines_thickness: int
    detector: str | None = None
    ransac_iterations: int | None = None
    elapsed_time: float | None = None
    success: bool | None = None


class ImageData(BaseModel):
    id: int = Field(alias="_id")
    session_id: str
    image: bytes
    size: int
    lines: int
    thickness: int


class ImageDataUpdate(BaseModel):
    id: int = Field(alias="_id")
    ransac_iterations: int
    detector: str
    success: bool
    elapsed_time: int


class ImageFindRequest(BaseModel):
    id: int = Field(alias="_id")
    ransac_iterations: int
    detector: str
