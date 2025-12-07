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
    create_user,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Đăng ký tài khoản mới.
    - Luôn tạo user với role = "user" (logic nằm trong create_user).
    """
    user = create_user(db, user_in)
    return user


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Đăng nhập bằng email + password.
    OAuth2PasswordRequestForm dùng field 'username' cho email.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai email hoặc mật khẩu",
        )

    access_token = create_access_token({"sub": str(user.id)})
    # Token schema có thể có token_type mặc định là "bearer"
    return Token(access_token=access_token)


@router.post("/forgot-password")
def forgot_password(
    payload: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Nhận email, nếu user tồn tại thì gửi mail chứa link reset password.
    Không lộ thông tin email có tồn tại hay không.
    """
    user = get_user_by_email(db, payload.email)

    if user:
        reset_token = create_password_reset_token(user.id)
        try:
            send_password_reset_email(user.email, reset_token)
        except Exception as e:
            # Không crash hệ thống vì lỗi SMTP, chỉ log ra
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
    Đặt lại mật khẩu bằng token nhận được qua email.
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
