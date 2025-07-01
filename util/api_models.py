from pydantic import BaseModel
from typing import Optional, Literal


# Common API
class CommonResponse(BaseModel):
    status: Literal["success", "error"]
    data: Optional[dict] = None
    error: Optional[dict] = None

# YOLO API
class IconDetectRequest(BaseModel):
    source: str
    conf: float = 0.05
    iou: float = 0.1

class IconDetectResponse(BaseModel):
    centers: list[list[float]]
    image: str

# Browser API
class GotoRequest(BaseModel):
    url: str

class MoveMouseRequest(BaseModel):
    x: float
    y: float
    step: Optional[int] = 1

class MovePathRequest(BaseModel):
    start_x: float  # normalized [0, 1]
    start_y: float
    end_x: float
    end_y: float
    steps: int = 20  # number of intermediate steps

class ClickRequest(BaseModel):
    x: float
    y: float

class GetScreenshotRequest(BaseModel):
    wait_time: float = 1.0

class ScrollRequest(BaseModel):
    delta_x: float
    delta_y: float

class TypeAtRequest(BaseModel):
    x: float
    y: float
    text: str

class WaitForLoadingRequest(BaseModel):
    wait_state: Literal["domcontentloaded", "networkidle", "load"] = "domcontentloaded"
    timeout: int = 30000