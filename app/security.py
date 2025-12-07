# app/security.py
import os
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Optional

from fastapi import HTTPException, status
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.models import User
from app.schemas import UserCreate

# ====== CẤU HÌNH JWT & TOKEN ======

# NÊN cấu hình trong .env:
# SECRET_KEY=chuoi_random_dai_va_bi_mat
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME_NOW")  # nhớ đổi trong .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")  # mặc định 1 ngày
)

# ====== PASSWORD HASHING ======

# Dùng pbkdf2_sha256 cho đơn giản, vẫn đủ an toàn cho đồ án
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash mật khẩu với một vài check cơ bản về độ dài.
    """
    if len(password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mật khẩu phải từ 6 ký tự trở lên.",
        )
    if len(password) > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mật khẩu không được dài hơn 128 ký tự.",
        )
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ====== JWT TOKEN (LOGIN & RESET PASSWORD) ======


def create_access_token(
    data: dict, expires_delta: Optional[timedelta] = None
) -> str:
    """
    Tạo access token (JWT) cho user đăng nhập.
    data thường sẽ chứa: {"sub": str(user.id)}
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_password_reset_token(user_id: int, expires_minutes: int = 30) -> str:
    """
    Tạo JWT dùng riêng cho việc reset password.
    scope = 'password_reset' để phân biệt với access_token đăng nhập.
    """
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode = {"sub": str(user_id), "scope": "password_reset", "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_password_reset_token(token: str) -> Optional[int]:
    """
    Giải token reset, trả về user_id nếu hợp lệ, ngược lại trả None.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("scope") != "password_reset":
            return None
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return int(user_id)
    except JWTError:
        return None


# ====== EMAIL / SMTP ======

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")          # ví dụ: yourgmail@gmail.com
SMTP_PASS = os.getenv("SMTP_PASS")          # app password (không phải mật khẩu thường)
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER or "no-reply@example.com")

# URL gốc của web, để build link trong email reset mật khẩu
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


def send_email(to_email: str, subject: str, body: str) -> None:
    """
    Hàm gửi email chung. Nếu chưa cấu hình SMTP thì in ra console (dev mode).
    """
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS):
        print("=== EMAIL (DEV MODE) ===")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(body)
        print("========================")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)


def send_password_reset_email(to_email: str, reset_token: str) -> None:
    """
    Gửi email chứa link đặt lại mật khẩu.
    """
    reset_link = f"{BASE_URL}/reset-password?token={reset_token}"
    subject = "DoctorX - Xác nhận quên mật khẩu"
    body = (
        "Bạn hoặc ai đó đã yêu cầu đặt lại mật khẩu tài khoản DoctorX.\\n\\n"
        f"Nhấn vào liên kết dưới đây để đặt lại mật khẩu trong 30 phút tới:\\n\\n"
        f"{reset_link}\\n\\n"
        "Nếu bạn không yêu cầu, hãy bỏ qua email này."
    )
    send_email(to_email, subject, body)


# ====== HÀM HỖ TRỢ VỀ USER ======


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Trả về User nếu email + password đúng, ngược lại trả None.
    """
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(db: Session, user_in: UserCreate) -> User:
    """
    Tạo user mới với role mặc định là 'user'.
    Dùng trong endpoint /auth/register.
    """
    # Kiểm tra email đã tồn tại chưa
    existing = get_user_by_email(db, user_in.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email đã tồn tại",
        )

    hashed = hash_password(user_in.password)
    db_user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=hashed,
        gender=user_in.gender,
        weight_kg=user_in.weight_kg,
        height_cm=user_in.height_cm,
        role="user",  # luôn là user, admin sẽ được set thủ công / qua route riêng
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# ====== DEVICE API KEY ======

import secrets


def generate_device_api_key() -> str:
    """
    Sinh API key cho device (dùng secrets để random).
    """
    return secrets.token_hex(16)
