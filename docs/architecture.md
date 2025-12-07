# Kiến trúc tổng quan hệ thống IoT Drinking Tracker (DoctorX)

## 1. Các lớp (layers) chính

1. **Device Layer (Things)**
   - Smart cup / thiết bị đo lượng nước (hiện tại giả lập bằng `mosquitto_pub` hoặc script).
   - Kết nối Wi-Fi / LAN tới MQTT broker.

2. **Gateway / Edge Layer**
   - Thành phần: `mqtt_worker.py`
   - Nhiệm vụ:
     - Subscribe topic: `doctorx/devices/+/telemetry`
     - Nhận telemetry từ MQTT, validate JSON
     - Gửi tiếp sang HTTP API `/ingest/telemetry` (FastAPI)
     - Đóng vai trò “IoT Gateway logic” (chuyển đổi protocol MQTT → HTTP/REST)

3. **IoT Platform / Backend Layer**
   - Thành phần:
     - FastAPI app (`app/main.py`)
     - `routes_auth.py`, `routes_devices.py`, `routes_telemetry.py`, `routes_water.py`
     - `db.py`, `models.py`, `schemas.py`, `security.py`
   - Nhiệm vụ:
     - Quản lý user, auth (JWT)
     - Quản lý device + API key
     - Nhận telemetry từ gateway và lưu vào DB (`Telemetry` table)
     - Tính toán tổng lượng nước, trạng thái cây, history theo ngày

4. **Data Storage Layer**
   - CSDL: SQLite (`iot_platform.db`)
   - Bảng chính:
     - `users`, `devices`, `telemetry`, `water_goal`, ...
   - ORM: SQLAlchemy

5. **Application / UI Layer**
   - Frontend HTML: `index.html`, `dashboard.html`, `garden/plant.html`, `water_detail.html`, ...
   - Giao tiếp backend qua REST API:
     - `/auth/*`, `/devices/*`, `/me/water/*`, `/devices/{id}/telemetry`

---

## 2. Mapping 8 mốc (protocol/layer)

1. **Data link layer**
   - Hiện tại: dùng hạ tầng LAN/Wi-Fi của máy tính (Ethernet/Wi-Fi).  
   - Thiết bị giả lập → coi như đang gắn vào mạng nội bộ qua driver OS.

2. **Network layer**
   - Sử dụng: **IP** (IPv4) trong LAN: `localhost`, `127.0.0.1`.
   - Broker MQTT, gateway, backend đều giao tiếp qua TCP/IP.

3. **Transport layer**
   - MQTT bên dưới dùng **TCP** (port 1883).
   - HTTP/REST (FastAPI) chạy trên **TCP** (port 8000).

4. **Session / Messaging / Application layer (IoT protocols)**
   - Giao thức messaging chính: **MQTT 3.1.1** (Paho client + Mosquitto).
   - Kiểu topic: `doctorx/devices/{device_id}/telemetry`.
   - Ứng dụng Cloud: **HTTP/REST + JSON** (FastAPI).

5. **Service layer**
   - Các dịch vụ backend:
     - Device service: đăng ký device, quản lý API key.
     - Telemetry ingest service: `/ingest/telemetry`.
     - Water analytics service: `/me/water/summary-today`, `/me/water/history`.
     - Auth service: `/auth/register`, `/auth/login`, `/auth/forgot-password`, `/auth/reset-password`.

6. **Security protocols**
   - JWT cho user auth (access token).
   - API key cho device auth (device_id + api_key).
   - MQTT broker:
     - `allow_anonymous false`
     - `password_file` với user: `iot_worker` + password
   - Password user: hash bằng `bcrypt`.
   - Reset password: signed token qua email (link `/reset-password?token=...`).

7. **API, SOA, OpenIoT / IoT–Cloud Convergence**
   - Backend expose REST APIs:
     - `/auth/*`, `/devices/*`, `/ingest/telemetry`, `/me/water/*`.
   - Gateway dùng HTTP API để đẩy data vào Cloud.
   - Kiến trúc kiểu **micro-service light** / SOA: mỗi nhóm route là một service logic.

8. **5 góc nhìn kiến trúc**
   - Physical / Deployment view
   - Logical view
   - Process / Data-flow view
   - Development / Module view
   - Security view

(Chi tiết từng view mô tả ở các mục tiếp theo.)
## 3. Physical / Deployment view

- **Máy dev / server local**:
  - Chạy Mosquitto broker (port 1883)
  - Chạy FastAPI (port 8000)
  - Chạy `mqtt_worker.py` (process riêng)
- **Device giả lập**:
  - Client MQTT (`mosquitto_pub` hoặc script Python) kết nối tới broker.
- Kết nối:
  - Device ↔ MQTT broker: TCP 1883
  - `mqtt_worker` ↔ broker: TCP 1883
  - `mqtt_worker` ↔ FastAPI: HTTP 8000
  - Browser ↔ FastAPI: HTTP 8000

## 4. Logical view

Các thành phần logic chính:

- **User Management**
  - Đăng ký, đăng nhập, JWT
  - Thay đổi mật khẩu, reset mật khẩu qua email

- **Device Management**
  - Tạo device, cấp API key
  - Device thuộc về 1 user (`owner_id`)

- **Telemetry Ingestion**
  - Nhận JSON telemetry từ gateway
  - Lưu vào bảng `telemetry`

- **Water Analytics**
  - Tính tổng ml trong ngày cho mỗi user
  - Tính số lần uống, lần nhiều nhất, trạng thái cây

- **Frontend UI**
  - Garden: tổng quan
  - Activities: chi tiết lần uống nước

## 5. Process / Data-flow view

### Flow 1: Uống nước từ thiết bị

1. User uống nước → thiết bị gửi MQTT:
   - Topic: `doctorx/devices/{device_id}/telemetry`
   - Payload JSON: `{ device_id, api_key, metric_type, value, payload }`
2. MQTT broker nhận message.
3. `mqtt_worker.py` (subscribed) nhận message:
   - Parse JSON
   - Gọi HTTP POST `/ingest/telemetry` với payload tương tự
4. FastAPI:
   - Xác thực `device_id + api_key`
   - Lưu record vào `telemetry`
   - Cập nhật số liệu cho màn hình Garden / Activities.

### Flow 2: User xem dashboard

1. User login → nhận JWT.
2. Frontend gọi:
   - `/me/water/summary-today` để lấy tổng ml, trạng thái cây.
   - `/devices/{device_id}/telemetry` để vẽ chart chi tiết.
3. Backend đọc DB và trả JSON.
4. Frontend render UI (cây, biểu đồ).

## 6. Development / Module view

- `app/main.py` – entrypoint FastAPI
- `app/db.py` – kết nối DB, SessionLocal
- `app/models.py` – ORM models (User, Device, Telemetry, ...)
- `app/schemas.py` – Pydantic schemas (request/response)
- `app/security.py` – JWT, password hashing, reset token, email
- `app/api/deps.py` – dependency: `get_db`, `get_current_user`
- `app/api/routes_auth.py` – auth & user routes
- `app/api/routes_devices.py` – device CRUD
- `app/api/routes_telemetry.py` – ingest + get telemetry
- `app/api/routes_water.py` – summary & history
- `app/api/routes_frontend.py` – serve HTML UI
- `mqtt_worker.py` – MQTT gateway (bridge MQTT → HTTP)

## 7. Security view

- **User side**
  - JWT access token, lưu ở frontend (localStorage / cookie).
  - Password hash bằng bcrypt.
  - Reset password flow với token có hạn.

- **Device side**
  - Mỗi device có `api_key` riêng, phải gửi cùng `device_id`.
  - Backend validate `device_id + api_key` trước khi nhận telemetry.

- **MQTT layer**
  - `listener 1883`
  - `allow_anonymous false`
  - `password_file C:/Test/mqtt_passwd.txt`
  - Worker dùng `iot_worker` + password để connect.

- **Network**
  - Dev: chạy trên localhost.
  - Khi triển khai thật: cần thêm HTTPS + MQTT over TLS (8883).
