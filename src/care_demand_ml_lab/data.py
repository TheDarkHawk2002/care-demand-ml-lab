from __future__ import annotations

import csv
from pathlib import Path

REQUIRED = {"request_id", "description", "service_area", "channel", "days_waiting", "open_items", "review_required"}


def load_records(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if len(rows) < 20:
        raise ValueError("At least 20 records are required.")
    ids = [r["request_id"].strip() for r in rows]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("request_id values must be non-empty and unique.")
    for index, row in enumerate(rows, start=2):
        if not row["description"].strip():
            raise ValueError(f"Empty description at CSV row {index}.")
        for name in ("days_waiting", "open_items"):
            try:
                value = int(row[name])
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer at CSV row {index}.") from exc
            if value < 0:
                raise ValueError(f"{name} cannot be negative at CSV row {index}.")
        if row["review_required"] not in {"0", "1"}:
            raise ValueError(f"review_required must be 0 or 1 at CSV row {index}.")
    if {r["review_required"] for r in rows} != {"0", "1"}:
        raise ValueError("Both label classes must occur in the data.")
    return rows
