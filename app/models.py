from datetime import datetime

from sqlmodel import Field, SQLModel


class ImageCreateRequest(SQLModel):
    square_size: int
    lines_qty: int
    lines_thickness: int


class ImageFindRequest(SQLModel):
    ransac_iterations: int
    detector: str


class SquareDetection(SQLModel, table=True):
    request_id: str | None = Field(default=None, primary_key=True)
    user_session: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    img_path: str | None = None
    square_size: int
    lines_qty: int
    lines_thickness: int
    detector: str | None = None
    ransac_iterations: int | None = None
    elapsed_time: float | None = None
    success: bool | None = None
