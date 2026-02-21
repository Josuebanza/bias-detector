"""Lightweight schema helpers (kept simple for hackathon).

If you want nicer docs fast, plug in drf-spectacular later.
"""

UPLOAD_RESPONSE_EXAMPLE = {
    "batch_id": "ab12cd34ef56",
    "inserted": 1234
}

ANALYZE_RESPONSE_KEYS = ["overall", "biases", "meta", "batch_id"]
