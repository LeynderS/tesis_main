from __future__ import annotations

from pathlib import Path

from bvg_core.config import (
    BVG_COLUMN_MAP,
    BVG_NUMERIC_COLUMNS,
    BVG_REQUIRED_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    TARGET_H,
    TEST_SIZE,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "BVG_features_svc_master.csv"
LOG_PATH = ROOT / "app" / "logs_trazabilidad.csv"
LAST_RUN_PATH = ROOT / "app" / "last_run.csv"
MODELS_DIR = ROOT / "models"

BVG_URL = "https://www.bolsadevaloresguayaquil.com/boletines/historicos/BVG_Acciones.xlsx"
BVG_TARGET_ISSUERS = [
    "BANCO GUAYAQUIL S.A.",
    "CORPORACION FAVORITA C.A.",
]

COMPANY_FILE_MAP = {
    "BANCO GUAYAQUIL S.A.": "BANCO_GUAYAQUIL_SA",
    "CORPORACION FAVORITA C.A.": "CORPORACION_FAVORITA_CA",
}

COMPANY_ABBR_MAP = {
    "BANCO GUAYAQUIL S.A.": "BG",
    "CORPORACION FAVORITA C.A.": "CF",
}

THEME_ACCENT_CLASSICAL = "#3b82f6"
THEME_ACCENT_QUANTUM = "#8b5cf6"

BVG_FALLBACK_MAX_DAYS = 4

HORIZONS = [5]

LOG_COLUMNS = [
    "id",
    "company",
    "model_family",
    "model_name",
    "horizonte",
    "fecha_t",
    "close_t",
    "y_pred",
    "proba_up",
    "status",
    "fecha_t5",
    "close_t5",
    "ret_fwd_h5",
    "Fecha_Datos_Usados",
    "Observacion_Datos",
]
