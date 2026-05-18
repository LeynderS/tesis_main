from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def safe_company(name: str) -> str:
    return name.replace(" ", "_").replace(".", "").replace("/", "-")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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
    "load_manifest",
    "require_file",
]
