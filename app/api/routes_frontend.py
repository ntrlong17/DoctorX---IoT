# app/api/routes_frontend.py
from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

router = APIRouter(tags=["frontend"])


@router.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app")


@router.get("/app", include_in_schema=False)
def serve_app():
    return FileResponse(FRONTEND_DIR / "index.html")


@router.get("/dashboard", include_in_schema=False)
def dashboard_page():
    return FileResponse(FRONTEND_DIR / "dashboard.html")


@router.get("/garden", include_in_schema=False)
def garden_page():
    return FileResponse(FRONTEND_DIR / "plant.html")


@router.get("/water_detail", include_in_schema=False)
def water_detail_page():
    return FileResponse(FRONTEND_DIR / "water_detail.html")


@router.get("/forgot-password", include_in_schema=False)
def forgot_password_page():
    return FileResponse(FRONTEND_DIR / "forgot_password.html")


@router.get("/reset-password", include_in_schema=False)
def reset_password_page():
    return FileResponse(FRONTEND_DIR / "reset_password.html")
