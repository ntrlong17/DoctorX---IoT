# app/api/routes_auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import User
from app.schemas import (
    UserCreate,
    UserOut,
    Token,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from app.security import (
    hash_password,
    authenticate_user,
    create_access_token,
    create_password_reset_token,
    send_password_reset_email,
    get_user_by_email,
    verify_password_reset_token,
)

router = APIRoter = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Đăng ký tài khoản mới.
    - Nhận JSON:
      {
        "email": "...",
        "password": "...",
        "gender": "male|female",
        "weight_kg": 60,
        "height_cm": 170
      }
    """
    # Kiểm tra trùng email
    existing = get_user_by_email(db, user_in.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email đã được đăng ký",
        )

    # Tạo user mới, role mặc định = "user"
    user = User(
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        full_name=user_in.full_name,
        gender=user_in.gender,
        weight_kg=user_in.weight_kg,
        height_cm=user_in.height_cm,
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Đăng nhập bằng email + password.
    - Frontend gửi form: username=email, password=...
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai email hoặc mật khẩu",
        )

    access_token = create_access_token({"sub": str(user.id)})
    return Token(access_token=access_token)


@router.post("/forgot-password")
def forgot_password(
    payload: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Nhận email, nếu tồn tại thì gửi mail reset (nếu cấu hình SMTP).
    Không tiết lộ email có tồn tại hay không.
    """
    user = get_user_by_email(db, payload.email)

    if user:
        reset_token = create_password_reset_token(user.id)
        try:
            send_password_reset_email(user.email, reset_token)
        except Exception as e:
            print("Lỗi gửi email reset mật khẩu:", e)

    return {
        "message": "Nếu email tồn tại trong hệ thống, đường dẫn đặt lại mật khẩu đã được gửi."
    }


@router.post("/reset-password")
def reset_password(
    data: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Đặt lại mật khẩu bằng token.
    """
    user_id = verify_password_reset_token(data.token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token đặt lại mật khẩu không hợp lệ hoặc đã hết hạn.",
        )

    user: User | None = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User không tồn tại.",
        )

    user.hashed_password = hash_password(data.new_password)
    db.commit()
    return {"message": "Mật khẩu đã được thay đổi thành công."}
