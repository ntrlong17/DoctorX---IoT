import requests
import time
import random
import sys

# Server đang chạy trên chính máy bạn, port 8000
BACKEND_URL = "https://hurried-paul-postmyxedematous.ngrok-free.dev"


def run_device(device_id: str, api_key: str, interval_sec: int = 5):
    while True:
        # Giả lập mỗi lần uống một lượng nước ngẫu nhiên
        volume = random.choice([100, 150, 200, 250, 300])  # ml

        payload = {
            "device_id": device_id,
            "api_key": api_key,
            "metric_type": "water_intake_ml",
            "value": float(volume),
            "payload": {
                "volume_ml": volume
            }
        }

        try:
            r = requests.post(f"{BACKEND_URL}/ingest/telemetry", json=payload, timeout=5)
            print(device_id, payload, "->", r.status_code, r.text)
        except Exception as e:
            print("Error:", e)

        time.sleep(interval_sec)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python device_sim.py <device_id> <api_key>")
        sys.exit(1)

    dev_id = sys.argv[1]
    api_key = sys.argv[2]
    run_device(dev_id, api_key)
