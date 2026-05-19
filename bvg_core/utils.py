from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def safe_company(name: str) -> str:
    return name.replace(" ", "_").replace(".", "").replace("/", "-")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of DataFrame CSV bytes, return first 8 hex chars."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()[:8]


def build_dataset_version(versioned_csv_path: Path) -> str:
    """Build version string ``vYYYYMMDD-<hash8>`` from filename and contents."""
    stem = versioned_csv_path.stem
    prefix = "BVG_features_svc_master_v"
    if stem.startswith(prefix):
        date_str = stem[len(prefix) :]
    else:
        import datetime as _dt

        mtime = versioned_csv_path.stat().st_mtime
        date_str = _dt.datetime.fromtimestamp(mtime).strftime("%Y%m%d")
    file_hash = sha256_file(versioned_csv_path)[:8]
    return f"v{date_str}-{file_hash}"


def write_dataset_manifest(
    path: Path,
    version: str,
    hash: str,
    row_count: int,
    date_range: str,
) -> None:
    """Write a minimal dataset manifest (optional, for audit)."""
    payload = {
        "dataset_version": version,
        "hash": hash,
        "row_count": row_count,
        "date_range": date_range,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path.as_posix()}")
    return path


def load_manifest(
    path: Path,
    *,
    required_fields: Iterable[str] | None = None,
) -> dict:
    require_file(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if required_fields:
        missing = [f for f in required_fields if f not in manifest]
        if missing:
            raise ValueError(
                "El manifest no contiene campos requeridos: " + ", ".join(missing)
            )
    return manifest


__all__ = [
    "safe_company",
    "sha256_file",
    "compute_dataset_hash",
    "build_dataset_version",
    "write_dataset_manifest",
    "load_manifest",
    "require_file",
]
