from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, EmailStr, validator


# ==== USER / AUTH ====


class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    role: str = "user"


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None


class UserOut(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True  # Pydantic v2 thay cho orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


# ==== DEVICE ====


class DeviceCreate(BaseModel):
    device_id: str
    name: Optional[str] = None


class DeviceOut(BaseModel):
    device_id: str
    name: Optional[str]
    api_key: str  # để bạn cấu hình cho thiết bị

    class Config:
        from_attributes = True


class DeviceCommandIn(BaseModel):
    command_type: str
    payload: Optional[Dict[str, Any]] = None


# ==== TELEMETRY ====


class TelemetryIn(BaseModel):
    device_id: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    metric_type: Optional[str] = Field(
        default="water_intake_ml",
        description="Loại metric, ví dụ: water_intake_ml",
    )
    value: Optional[float] = Field(
        default=None,
        description="Giá trị chính, dùng ml cho water_intake",
    )
    payload: Optional[Dict[str, Any]] = None

    @validator("value")
    def validate_value(cls, v, values):
        # Chỉ check chặt cho nước uống
        metric = values.get("metric_type")
        if metric in ("water_intake_ml", "water_intake"):
            if v is None:
                raise ValueError("value is required for water_intake")
            if v <= 0:
                raise ValueError("value must be > 0")
            if v > 2000:
                raise ValueError(
                    "value too large for a single drink (max 2000 ml)"
                )
        return v


class TelemetryOut(BaseModel):
    ts: datetime
    metric_type: Optional[str]
    value: Optional[float]
    payload: Optional[dict]

    class Config:
        from_attributes = True


# ==== WATER / DASHBOARD ====


class WaterSummaryOut(BaseModel):
    date: str
    total_ml: float
    percent: float
    plant_state: str   # dry / growing / healthy / bloom
    time_slot: str     # morning / lunch / afternoon / night
    image: str         # tên file ảnh, VD: plant_dry_afternoon.png
    target_ml: float


class WaterHistoryDay(BaseModel):
    date: str          # "2025-12-05"
    total_ml: float    # tổng ml trong ngày
    percent: float     # % so với target (ví dụ 2000 ml)


class WaterHistoryOut(BaseModel):
    days: List[WaterHistoryDay]


class DashboardOut(BaseModel):
    today: WaterSummaryOut          # summary hôm nay
    last_7_days: List[WaterHistoryDay]
