# app/api/routes_water.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, datetime, timedelta
from typing import List, Optional

from app.api.deps import get_db, get_current_user
from app.models import Telemetry, Device, User
from app.schemas import (
    WaterSummaryOut,
    WaterHistoryOut,
    WaterHistoryDay,
    DashboardOut,
)

router = APIRouter(prefix="/me/water", tags=["water"])


# ==== HELPER FUNCTIONS CHO NƯỚC ====


def calc_daily_water_target(user: User) -> float:
    if not user.weight_kg:
        return 2000.0

    if user.gender == "male":
        return user.weight_kg * 35
    else:
        return user.weight_kg * 31


def classify_plant_state(total_ml: float) -> str:
    if total_ml < 500:
        return "dry"
    elif total_ml < 1200:
        return "growing"
    elif total_ml < 2000:
        return "healthy"
    else:
        return "bloom"


def get_time_slot(now: Optional[datetime] = None) -> str:
    if now is None:
        now = datetime.utcnow() + timedelta(hours=7)

    h = now.hour
    if 5 <= h < 11:
        return "morning"
    elif 11 <= h < 14:
        return "lunch"
    elif 14 <= h < 18:
        return "afternoon"
    else:
        return "night"


def get_plant_image(plant_state: str, time_slot: str) -> str:
    return f"plant_{plant_state}_{time_slot}.png"


# ==== ROUTES ====


@router.get("/summary-today", response_model=WaterSummaryOut)
def get_today_water_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    devices = db.query(Device).filter(Device.owner_id == current_user.id).all()
    today = (datetime.utcnow() + timedelta(hours=7)).date()
    daily_target = calc_daily_water_target(current_user)

    if not devices:
        plant_state = classify_plant_state(0.0)
        time_slot = get_time_slot()
        return WaterSummaryOut(
            date=today.isoformat(),
            total_ml=0.0,
            percent=0.0,
            plant_state=plant_state,
            time_slot=time_slot,
            image=get_plant_image(plant_state, time_slot),
            target_ml=daily_target,
        )

    device_ids = [d.device_id for d in devices]

    start = datetime(today.year, today.month, today.day) - timedelta(hours=7)
    end = start + timedelta(days=1)

    total = (
        db.query(func.sum(Telemetry.value))
        .filter(
            Telemetry.device_id.in_(device_ids),
            Telemetry.metric_type == "water_intake_ml",
            Telemetry.ts >= start,
            Telemetry.ts < end,
        )
        .scalar()
        or 0.0
    )
    total = float(total)
    percent = min(total / daily_target * 100.0, 100.0)

    plant_state = classify_plant_state(total)
    time_slot = get_time_slot()
    image = get_plant_image(plant_state, time_slot)

    return WaterSummaryOut(
        date=today.isoformat(),
        total_ml=total,
        percent=percent,
        plant_state=plant_state,
        time_slot=time_slot,
        image=image,
        target_ml=daily_target,
    )


@router.get("/history", response_model=WaterHistoryOut)
def get_water_history(
    days: int = 7,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if days < 1:
        days = 1
    if days > 30:
        days = 30

    devices = db.query(Device).filter(Device.owner_id == current_user.id).all()
    today = (datetime.utcnow() + timedelta(hours=7)).date()
    daily_target = calc_daily_water_target(current_user)

    if not devices:
        history_days: List[WaterHistoryDay] = []
        for i in range(days - 1, -1, -1):
            d = today - timedelta(days=i)
            history_days.append(
                WaterHistoryDay(
                    date=d.isoformat(),
                    total_ml=0.0,
                    percent=0.0,
                )
            )
        return WaterHistoryOut(days=history_days)

    device_ids = [d.device_id for d in devices]
    history_days: List[WaterHistoryDay] = []

    for i in range(days - 1, -1, -1):
        d: date = today - timedelta(days=i)

        start = datetime(d.year, d.month, d.day) - timedelta(hours=7)
        end = start + timedelta(days=1)

        total = (
            db.query(func.sum(Telemetry.value))
            .filter(
                Telemetry.device_id.in_(device_ids),
                Telemetry.metric_type == "water_intake_ml",
                Telemetry.ts >= start,
                Telemetry.ts < end,
            )
            .scalar()
            or 0.0
        )
        total = float(total)
        percent = min(total / daily_target * 100.0, 100.0)

        history_days.append(
            WaterHistoryDay(
                date=d.isoformat(),
                total_ml=total,
                percent=percent,
            )
        )

    return WaterHistoryOut(days=history_days)


@router.get("/dashboard", response_model=DashboardOut)
def get_water_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    today_summary = get_today_water_summary(db=db, current_user=current_user)
    history = get_water_history(days=7, db=db, current_user=current_user)

    return DashboardOut(
        today=today_summary,
        last_7_days=history.days,
    )
