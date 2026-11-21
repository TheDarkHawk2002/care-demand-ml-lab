from __future__ import annotations

import csv
import random
from pathlib import Path

AREAS = ["appointments", "home_support", "transport", "equipment"]
CHANNELS = ["phone", "web", "referral"]
LOW = ["routine update", "schedule question", "document request", "general information"]
HIGH = ["missed visit repeated delay", "equipment unavailable urgent follow-up", "multiple open requests overdue", "transport delay affects scheduled service"]


def generate_rows(count: int, seed: int = 41) -> list[dict[str, str | int]]:
    if count < 20:
        raise ValueError("Use at least 20 rows to allow a meaningful holdout split.")
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        urgent = rng.random() < 0.34
        waiting = rng.randint(4, 24) if urgent else rng.randint(0, 10)
        open_items = rng.randint(2, 6) if urgent else rng.randint(0, 3)
        description = rng.choice(HIGH if urgent else LOW)
        label = int(urgent if rng.random() > 0.08 else not urgent)
        rows.append({"request_id": f"SYN-{i + 1:04d}", "description": description,
                     "service_area": rng.choice(AREAS), "channel": rng.choice(CHANNELS),
                     "days_waiting": waiting, "open_items": open_items, "review_required": label})
    return rows


def write_csv(path: str | Path, count: int = 360, seed: int = 41) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(count, seed)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target
