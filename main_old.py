from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta, date
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Float, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.types import JSON
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.db import SessionLocal, Base, engine
from app.models import User, Device, Telemetry
from app.schemas import (
    UserCreate,
    UserOut,
    Token,
    DeviceCreate,
    DeviceOut,
    TelemetryIn,
    TelemetryOut,
    WaterSummaryOut,
    WaterHistoryDay,
    WaterHistoryOut,
    DashboardOut,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from app.security import (
    SECRET_KEY,
    ALGORITHM,
    get_password_hash,
    verify_password,
    create_access_token,
    create_password_reset_token,
    verify_password_reset_token,
    send_password_reset_email,
    get_user_by_email,
    authenticate_user,
    generate_device_api_key,
)

Base.metadata.create_all(bind=engine)
# ======================
# BẢO MẬT (PASSWORD & JWT)
# ======================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ======================
# DEPENDENCY & HELPER
# ======================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def calc_daily_water_target(user: User) -> float:
    """
    Tính lượng nước cần uống hằng ngày dựa trên:
    - giới tính (gender)
    - cân nặng (kg)

    Công thức:
        Nam: 35 ml × kg
        Nữ: 31 ml × kg
    """

    if not user.weight_kg:
        return 2000.0   # fallback

    if user.gender == "male":
        return user.weight_kg * 35
    else:
        return user.weight_kg * 31


def classify_plant_state(total_ml: float) -> str:
    """Phân loại trạng thái cây dựa trên lượng nước trong ngày."""
    if total_ml < 500:
        return "dry"
    elif total_ml < 1200:
        return "growing"
    elif total_ml < 2000:
        return "healthy"
    else:
        return "bloom"


def get_time_slot(now: Optional[datetime] = None) -> str:
    """
    Chia time slot trong ngày.
    Tạm lấy giờ VN = UTC + 7 (đang chạy local nên chấp nhận được).
    """
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
    """
    Sinh tên file ảnh cho frontend.
    Bạn đã có đủ file:
    plant_dry_morning.png, plant_dry_lunch.png, ...
    nên cứ map thẳng như này.
    """
    return f"plant_{plant_state}_{time_slot}.png"


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user


# ======================
# FASTAPI APP
# ======================
app = FastAPI(title="ERA-like IoT Platform")

# Serve giao diện web người dùng ở /app
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ---------- AUTH (USER / APP) ----------
@app.post("/auth/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = get_user_by_email(db, user_in.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email đã được đăng ký")

    user = User(
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        gender=user_in.gender,
        weight_kg=user_in.weight_kg,
        height_cm=user_in.height_cm,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user



@app.post("/auth/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # OAuth2PasswordRequestForm dùng field "username" cho email
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        # Trả HTTP 401 giống thực tế
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai email hoặc mật khẩu"
        )
    access_token = create_access_token({"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/forgot-password")
def forgot_password(
    payload: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Người dùng nhập email. Nếu tồn tại, tạo token reset password và gửi link qua email.
    Luôn trả cùng một message để tránh lộ thông tin tài khoản.
    """
    user = get_user_by_email(db, payload.email)

    if user:
        reset_token = create_password_reset_token(user.id)
        try:
            send_password_reset_email(user.email, reset_token)
        except Exception as e:
            # log để debug, không tiết lộ chi tiết cho client
            print("Lỗi gửi email reset mật khẩu:", e)

    return {
        "message": "Nếu email tồn tại trong hệ thống, đường dẫn đặt lại mật khẩu đã được gửi."
    }


@app.post("/auth/reset-password")
def reset_password(
    data: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Nhận token + mật khẩu mới, đổi mật khẩu cho user tương ứng.
    """
    user_id = verify_password_reset_token(data.token)
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="Token đặt lại mật khẩu không hợp lệ hoặc đã hết hạn."
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="User không tồn tại.")

    user.hashed_password = get_password_hash(data.new_password)
    db.commit()
    return {"message": "Mật khẩu đã được thay đổi thành công."}


# ---------- DEVICE REGISTRY ----------
@app.post("/devices", response_model=DeviceOut)
def create_device(
    device_in: DeviceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = db.query(Device).filter(Device.device_id == device_in.device_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Device ID already exists")

    api_key = generate_device_api_key()
    dev = Device(
        device_id=device_in.device_id,
        name=device_in.name,
        owner_id=current_user.id,
        api_key=api_key,
    )
    db.add(dev)
    db.commit()
    db.refresh(dev)
    return dev


@app.get("/devices", response_model=List[DeviceOut])
def list_devices(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    devs = db.query(Device).filter(Device.owner_id == current_user.id).all()
    return devs


# ---------- TELEMETRY INGEST (CHO THIẾT BỊ) ----------
@app.post("/ingest/telemetry")
def ingest_telemetry(
    data: TelemetryIn,
    db: Session = Depends(get_db),
):
    # Thiết bị không dùng JWT, dùng device_id + api_key
    dev = db.query(Device).filter(Device.device_id == data.device_id).first()
    if not dev:
        raise HTTPException(status_code=400, detail="Unknown device_id")

    if dev.api_key != data.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    row = Telemetry(
        device_id=data.device_id,
        metric_type=data.metric_type,
        value=data.value,
        payload=data.payload,
    )
    db.add(row)
    db.commit()
    return {"status": "ok"}


# ---------- TELEMETRY QUERY (CHO USER/APP) ----------
@app.get("/devices/{device_id}/telemetry", response_model=List[TelemetryOut])
def get_telemetry(
    device_id: str,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dev = db.query(Device).filter(Device.device_id == device_id).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Device not found")

    if dev.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your device")

    rows = (
        db.query(Telemetry)
        .filter(Telemetry.device_id == device_id)
        .order_by(Telemetry.ts.desc())
        .limit(limit)
        .all()
    )
    return rows


# ---------- API SUMMARY HÔM NAY ----------
@app.get("/me/water/summary-today", response_model=WaterSummaryOut)
def get_today_water_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Lấy tất cả device của user
    devices = db.query(Device).filter(Device.owner_id == current_user.id).all()
    today = (datetime.utcnow() + timedelta(hours=7)).date()
    daily_target = calc_daily_water_target(current_user)
    if not devices:
        # Không có thiết bị nào
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

    # Khoảng thời gian hôm nay (theo VN time tạm tính = UTC+7)
    start = datetime(today.year, today.month, today.day) - timedelta(hours=7)
    end = start + timedelta(days=1)

    # Tổng value telemetry "water_intake_ml" trong ngày
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
    daily_target = calc_daily_water_target(current_user)
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
    target_ml=daily_target
)



# ---------- API LỊCH SỬ 7 NGÀY ----------
@app.get("/me/water/history", response_model=WaterHistoryOut)
def get_water_history(
    days: int = 7,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Trả về tổng ml theo từng ngày trong N ngày gần nhất (mặc định 7).
    """
    if days < 1:
        days = 1
    if days > 30:
        days = 30

    devices = db.query(Device).filter(Device.owner_id == current_user.id).all()
    today = (datetime.utcnow() + timedelta(hours=7)).date()
    daily_target = calc_daily_water_target(current_user)

    if not devices:
        # Không có device: trả list ngày với 0 ml
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

    # duyệt từ ngày cũ -> mới để dễ vẽ chart
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


# ---------- API TỔNG HỢP DASHBOARD ----------
@app.get("/me/water/dashboard", response_model=DashboardOut)
def get_water_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    API tổng hợp cho frontend dashboard:
    - today: summary hôm nay
    - last_7_days: lịch sử 7 ngày
    """
    today_summary: WaterSummaryOut = get_today_water_summary(
        db=db, current_user=current_user
    )
    history: WaterHistoryOut = get_water_history(
        days=7, db=db, current_user=current_user
    )

    return DashboardOut(
        today=today_summary,
        last_7_days=history.days,
    )


# ================== FRONTEND ROUTES ==================
@app.get("/app", include_in_schema=False)
async def serve_app():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        # Debug cho dễ thấy lỗi nếu sai đường dẫn
        raise HTTPException(
            status_code=500,
            detail=f"index.html not found at {index_file}"
        )
    return FileResponse(index_file)


@app.get("/water-detail", include_in_schema=False)
async def serve_water_detail():
    return FileResponse(FRONTEND_DIR / "water_detail.html")


@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    dash_file = FRONTEND_DIR / "dashboard.html"
    if not dash_file.exists():
        raise HTTPException(
            status_code=500,
            detail=f"dashboard.html not found at {dash_file}"
        )
    return FileResponse(dash_file)


@app.get("/garden", include_in_schema=False)
async def serve_garden():
    garden_file = FRONTEND_DIR / "plant.html"
    if not garden_file.exists():
        raise HTTPException(
            status_code=500,
            detail=f"plant.html not found at {garden_file}"
        )
    return FileResponse(garden_file)


@app.get("/forgot-password", include_in_schema=False)
def serve_forgot_password():
    return FileResponse(FRONTEND_DIR / "forgot_password.html")


@app.get("/reset-password", include_in_schema=False)
def serve_reset_password():
    return FileResponse(FRONTEND_DIR / "reset_password.html")


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app")

@app.get("/water_detail")
def water_detail_page():
    return FileResponse("frontend/water_detail.html")