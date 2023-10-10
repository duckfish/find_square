from typing import Optional

from fastapi import Form
from pydantic import BaseModel, Field


class UserData(BaseModel):
    id: int = Field(alias="_id")
    session_id: Optional[str] = Field(None)


class ImageData(BaseModel):
    id: int = Field(alias="_id")
    session_id: str
    image: bytes


class ImageDataUpdate(BaseModel):
    id: int = Field(alias="_id")
    image_result: bytes
    elapsed_time: int


class ImageParams(BaseModel):
    square_size: int = Form()
    lines_numb: int = Form()
    line_thickness: int = Form()
