# app/api/routes_admin.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.models import User
from app.schemas import UserOut  # schema trả về thông tin user

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Chỉ admin mới xem được danh sách toàn bộ user trong hệ thống.
    """
    users = db.query(User).all()
    return users
