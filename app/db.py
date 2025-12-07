# db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# URL mặc định dùng SQLite trong file iot_platform.db
DEFAULT_DB_URL = "sqlite:///./iot_platform.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Nếu dùng SQLite thì mới cần connect_args
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Tạo engine
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args
)

# SessionLocal dùng cho Depends(get_db)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base để khai báo model
Base = declarative_base()
