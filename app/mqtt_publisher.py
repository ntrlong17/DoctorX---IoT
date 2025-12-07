# app/mqtt_publisher.py
import json
import os
from uuid import uuid4

import paho.mqtt.client as mqtt

MQTT_BROKER_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_BASE_TOPIC = os.getenv("MQTT_BASE_TOPIC", "doctorx")


def publish_device_command(device_id: str, command: dict, qos: int = 1) -> None:
    """
    Publish lệnh xuống thiết bị qua MQTT.
    Topic: doctorx/devices/{device_id}/commands
    """
    topic = f"{MQTT_BASE_TOPIC}/devices/{device_id}/commands"

    client_id = f"doctorx_cmd_{device_id}_{uuid4().hex[:8]}"
    client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)

    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
    client.loop_start()
    client.publish(topic, json.dumps(command), qos=qos, retain=False)
    client.loop_stop()
    client.disconnect()
