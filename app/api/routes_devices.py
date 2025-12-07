# app/api/routes_devices.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_db, get_current_user
from app.models import Device, User
from app.schemas import DeviceCreate, DeviceOut, DeviceCommandIn
from app.mqtt_publisher import publish_device_command
from app.security import generate_device_api_key

router = APIRouter(prefix="/devices", tags=["devices"])


@router.post("", response_model=DeviceOut)
def create_device(
    device_in: DeviceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = db.query(Device).filter(Device.device_id == device_in.device_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Device ID đã tồn tại")

    api_key = generate_device_api_key()
    device = Device(
        device_id=device_in.device_id,
        name=device_in.name,
        owner_id=current_user.id,
        api_key=api_key,
    )
    db.add(device)
    db.commit()
    db.refresh(device)
    return device


@router.get("", response_model=list[DeviceOut])
def list_devices(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    devices = db.query(Device).filter(Device.owner_id == current_user.id).all()
    return devices

@router.post("/{device_id}/commands")
def send_device_command(
    device_id: str,
    cmd: DeviceCommandIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Gửi lệnh xuống device qua MQTT:
    - Kiểm tra device có thuộc về current_user không
    - Publish command JSON lên topic doctorx/devices/{device_id}/commands
    """
    device = (
        db.query(Device)
        .filter(
            Device.device_id == device_id,
            Device.owner_id == current_user.id,
        )
        .first()
    )
    if not device:
        raise HTTPException(status_code=404, detail="Device không tồn tại")

    command = {
        "device_id": device.device_id,
        "command_type": cmd.command_type,
        "payload": cmd.payload or {},
    }

    publish_device_command(device.device_id, command)
    return {"status": "queued"}
