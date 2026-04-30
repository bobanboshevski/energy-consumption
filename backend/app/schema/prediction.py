from pydantic import BaseModel
from typing import Optional


class ForecastPoint(BaseModel):
    date: str
    predicted_demand: float
    demand_category: str
    temp_max: Optional[float]
    temp_min: Optional[float]
    daylight_duration: Optional[float]


class HistoricalPoint(BaseModel):
    Date: str
    energy_demand: float
    temp_max: Optional[float]
    temp_min: Optional[float]
    daylight_duration: Optional[float]
