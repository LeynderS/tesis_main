from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# Ecuador timezone (UTC-5)
ECUADOR_OFFSET = timedelta(hours=-5)


def read_last_run(path: Path) -> datetime | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "last_run_utc" not in df.columns:
            return None
        ts = pd.to_datetime(df["last_run_utc"].iloc[0], errors="coerce")
        return ts if pd.notna(ts) else None
    except Exception:
        return None


def write_last_run(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"last_run_utc": [datetime.now(timezone.utc).isoformat()]}).to_csv(
        path, index=False
    )


def _to_local(dt: datetime) -> datetime:
    """Convert UTC datetime to Ecuador local time (UTC-5)."""
    ecuador_tz = timezone(timedelta(hours=-5))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ecuador_tz)


def format_last_run(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    local_dt = _to_local(dt)
    meses = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ]
    return (
        f"{local_dt.day} de {meses[local_dt.month - 1]} del {local_dt.year} "
        f"a las {local_dt.hour:02d}:{local_dt.minute:02d}"
    )
