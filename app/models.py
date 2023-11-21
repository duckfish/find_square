from pydantic import BaseModel, Field


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


class ImageCreateRequest(BaseModel):
    id: int = Field(alias="_id")
    session_id: str
    square_size: int
    lines_numb: int
    line_thickness: int


class ImageFindRequest(BaseModel):
    id: int = Field(alias="_id")
    ransac_iterations: int
    detector: str
