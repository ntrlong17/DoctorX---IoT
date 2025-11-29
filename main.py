import os
import smtplib
from email.message import EmailMessage
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.types import JSON
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from fastapi.responses import RedirectResponse
from fastapi import HTTPException

# ======================
# CẤU HÌNH DB
# ======================
DEFAULT_DB_URL = "sqlite:///./iot_platform.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Nếu dùng SQLite thì mới cần connect_args
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ======================
# BẢO MẬT (PASSWORD & JWT)
# ======================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

SECRET_KEY = "super_secret_key_change_me"  # đổi khi deploy thật
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 ngày

# ======================
# SMTP / EMAIL CONFIG
# ======================
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")      # ví dụ: yourgmail@gmail.com
SMTP_PASS = os.getenv("SMTP_PASS")      # app password (không phải mật khẩu thường)
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER or "no-reply@example.com")

# URL gốc của web, để build link trong email
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


def send_password_reset_email(to_email: str, reset_token: str):
    """
    Gửi email chứa link đặt lại mật khẩu.
    Nếu chưa cấu hình SMTP, chỉ in link ra console (dev mode).
    """
    reset_link = f"{BASE_URL}/reset-password?token={reset_token}"

    # Nếu chưa cấu hình SMTP đầy đủ -> in link ra console cho dev
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS):
        print("=== PASSWORD RESET LINK (DEV) ===")
        print(reset_link)
        print("=================================")
        return

    msg = EmailMessage()
    msg["Subject"] = "DoctorX - Xác nhận quên mật khẩu"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg.set_content(
        f"Bạn hoặc ai đó đã yêu cầu đặt lại mật khẩu tài khoản DoctorX.\n\n"
        f"Nhấn vào liên kết dưới đây để đặt lại mật khẩu trong 30 phút tới:\n\n"
        f"{reset_link}\n\n"
        f"Nếu bạn không yêu cầu, hãy bỏ qua email này."
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)


# ======================
# MODEL DB
# ======================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    devices = relationship("Device", back_populates="owner")


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    api_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)  # 1 = active, 0 = inactive

    owner = relationship("User", back_populates="devices")
    telemetry = relationship("Telemetry", back_populates="device")


class Telemetry(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, ForeignKey("devices.device_id"), index=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    metric_type = Column(String, nullable=True)   # ví dụ: "water_intake_ml"
    value = Column(Float, nullable=True)          # ví dụ: 250.0
    payload = Column(JSON, nullable=True)         # raw data (JSON)

    device = relationship("Device", back_populates="telemetry")


Base.metadata.create_all(bind=engine)

# ======================
# SCHEMA (Pydantic)
# ======================
class UserCreate(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class DeviceCreate(BaseModel):
    device_id: str
    name: Optional[str] = None


class DeviceOut(BaseModel):
    device_id: str
    name: Optional[str]
    api_key: str  # để bạn cấu hình cho thiết bị

    class Config:
        orm_mode = True


class TelemetryIn(BaseModel):
    device_id: str
    api_key: str
    metric_type: Optional[str] = None
    value: Optional[float] = None
    payload: Optional[dict] = None


class TelemetryOut(BaseModel):
    ts: datetime
    metric_type: Optional[str]
    value: Optional[float]
    payload: Optional[dict]

    class Config:
        orm_mode = True


# ---- Forgot / Reset password ----
class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


# ======================
# DEPENDENCY & HELPER
# ======================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password: str) -> str:
    # bcrypt giới hạn 72 bytes, mình giới hạn 72 ký tự cho đơn giản
    if len(password) > 72:
        raise HTTPException(
            status_code=400,
            detail="Mật khẩu không được dài hơn 72 ký tự."
        )
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_password_reset_token(user_id: int, expires_minutes: int = 30) -> str:
    """
    Tạo JWT riêng cho việc reset password.
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


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def generate_device_api_key() -> str:
    return secrets.token_hex(16)


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


@app.get("/forgot-password", include_in_schema=False)
def serve_forgot_password():
    return FileResponse(FRONTEND_DIR / "forgot_password.html")


@app.get("/reset-password", include_in_schema=False)
def serve_reset_password():
    return FileResponse(FRONTEND_DIR / "reset_password.html")

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app")
