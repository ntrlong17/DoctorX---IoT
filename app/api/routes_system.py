from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text  # <-- THÊM DÒNG NÀY

from app.api.deps import get_db

router = APIRouter(tags=["system"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """
    Healthcheck đơn giản:
    - Nếu kết nối được DB -> status = ok
    - Dùng cho monitoring / kiểm tra nhanh.
    """
    try:
        # SQLAlchemy 1.4+/2.0 phải dùng text()
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_ok = False
        # nếu muốn, có thể log error e ở đây

    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "time": datetime.utcnow().isoformat() + "Z",
    }
