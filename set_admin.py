# set_admin.py
from app.db import SessionLocal
from app.models import User

EMAIL = "ntrlong17@gmail.com"  # <-- ĐỔI thành email bạn muốn làm admin

db = SessionLocal()

user = db.query(User).filter(User.email == EMAIL).first()
if not user:
    print("Không tìm thấy user với email:", EMAIL)
else:
    user.role = "admin"
    db.commit()
    db.refresh(user)
    print("Đã set admin cho:", user.email, "role =", user.role)

db.close()
