# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from dotenv import load_dotenv

from app.db import Base, engine

# Import từng router
from app.api.routes_auth import router as auth_router
from app.api.routes_devices import router as devices_router
from app.api.routes_telemetry import router as telemetry_router
from app.api.routes_water import router as water_router
from app.api.routes_frontend import router as frontend_router
from app.api.routes_system import router as system_router
from app.api.routes_admin import router as admin_router  # <- ADMIN ROUTER

# ==== Load .env ====
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ==== Tạo DB schema nếu chưa có ====
Base.metadata.create_all(bind=engine)

# ==== App chính ====
app = FastAPI(title="DoctorX IoT Water Tracker")

# ==== Static (nếu bạn có css/js riêng) ====
FRONTEND_DIR = ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ==== Gắn routers ====
app.include_router(auth_router)
app.include_router(devices_router)
app.include_router(telemetry_router)
app.include_router(water_router)
app.include_router(frontend_router)
app.include_router(system_router)
app.include_router(admin_router)  # <- dùng admin_router chứ không phải routes_admin.router
