from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List

class DetectRequest(BaseModel):
    url: Optional[HttpUrl] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    ts: Optional[str] = Field(None, description="ISO8601 timestamp")

class DetectResponse(BaseModel):
    label: str
    prob: float
    uncertainty: float
    grad_cam_url: Optional[str]
    event_id: str

class PlanRequest(BaseModel):
    event_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    label: str
    prob: float
    uncertainty: float
    objectives: str = "Rapid triage and recommended actions"

class PlanResponse(BaseModel):
    plan: str
    used_provider: str

class CalibrationPoint(BaseModel):
    prob_bin_center: float
    accuracy: float

class CalibrationReport(BaseModel):
    ece: float
    points: List[CalibrationPoint]
