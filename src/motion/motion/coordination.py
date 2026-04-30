#!/usr/bin/env python3

import json
from typing import Any

from std_msgs.msg import String


def encode_payload(payload: dict[str, Any]) -> String:
    message = String()
    message.data = json.dumps(payload, sort_keys=True)
    return message


def decode_payload(message: String) -> dict[str, Any] | None:
    try:
        payload = json.loads(message.data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
