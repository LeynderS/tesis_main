from __future__ import annotations

from bvg_core.config import (
    BVG_COLUMN_MAP,
    BVG_NUMERIC_COLUMNS,
    BVG_REQUIRED_COLUMNS,
    COMPANY_FILE_MAP,
    DATA_PATH,
    HORIZONS,
    LAST_RUN_PATH,
    LOG_PATH,
    MODELS_DIR,
    REQUIRED_RAW_COLUMNS,
    ROOT,
    TARGET_H,
    TEST_SIZE,
    COMPANIES as BVG_TARGET_ISSUERS,
)

BVG_URL = "https://www.bolsadevaloresguayaquil.com/boletines/historicos/BVG_Acciones.xlsx"

COMPANY_ABBR_MAP = {
    "BANCO GUAYAQUIL S.A.": "BG",
    "CORPORACION FAVORITA C.A.": "CF",
}

THEME_ACCENT_CLASSICAL = "#3b82f6"
THEME_ACCENT_QUANTUM = "#8b5cf6"

BVG_FALLBACK_MAX_DAYS = 4

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
