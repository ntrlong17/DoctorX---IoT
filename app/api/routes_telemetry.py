# app/api/routes_telemetry.py
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.models import Device, Telemetry, User
from app.schemas import TelemetryIn, TelemetryOut

router = APIRouter(tags=["telemetry"])


@router.post("/ingest/telemetry")
def ingest_telemetry(
    payload: TelemetryIn,
    db: Session = Depends(get_db),
):
    device = (
        db.query(Device)
        .filter(
            Device.device_id == payload.device_id,
            Device.api_key == payload.api_key,
            Device.is_active == 1,
        )
        .first()
    )
    if not device:
        raise HTTPException(status_code=401, detail="Device không hợp lệ")

    # Validation bổ sung ở tầng business
    if payload.metric_type in ("water_intake_ml", "water_intake"):
        if payload.value is None:
            raise HTTPException(
                status_code=400,
                detail="value is required for water_intake",
            )
        if payload.value <= 0 or payload.value > 2000:
            raise HTTPException(
                status_code=400,
                detail="value must be in range (0, 2000] ml",
            )

    telemetry = Telemetry(
        device_id=payload.device_id,
        ts=datetime.utcnow(),
        metric_type=payload.metric_type,
        value=payload.value,
        payload=payload.payload,
    )
    db.add(telemetry)
    db.commit()
    return {"status": "ok"}

@router.get("/devices/{device_id}/telemetry", response_model=List[TelemetryOut])
def get_device_telemetry(
    device_id: str,
    limit: int = 300,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    API cho frontend water_detail.html:
    trả danh sách telemetry của 1 device, để lọc theo ngày & vẽ biểu đồ.
    """

    # Kiểm tra device có tồn tại và thuộc về user hiện tại không
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device không tồn tại")

    if device.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Bạn không có quyền xem device này")

    # Giới hạn limit cho an toàn
    if limit < 1:
        limit = 1
    if limit > 1000:
        limit = 1000

    rows = (
        db.query(Telemetry)
        .filter(Telemetry.device_id == device_id)
        .order_by(Telemetry.ts.desc())
        .limit(limit)
        .all()
    )

    # TelemetryOut có orm_mode = True nên cứ trả thẳng rows
    return rows
