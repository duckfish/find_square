from typing import Optional

from bson import Binary
from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field


class UserData(BaseModel):
    id: int = Field(alias="_id")
    session_id: Optional[str] = Field(None)


class ImageData(UserData):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: bytes


class ImageParams(BaseModel):
    square_size: int = Form()
    lines_numb: int = Form()
    line_thickness: int = Form()
