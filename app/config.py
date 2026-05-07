from __future__ import annotations

from pathlib import Path

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

BVG_REQUIRED_COLUMNS = {
    "FECHA NEGOCIACIÓN",
    "EMISOR",
    "PRECIO",
    "NÚMERO DE ACCIONES",
    "VALOR EFECTO",
}

BVG_COLUMN_MAP = {
    "FECHA NEGOCIACIÓN": "fecha",
    "EMISOR": "empresa",
    "PRECIO": "precio",
    "NÚMERO DE ACCIONES": "numero_acciones",
    "VALOR EFECTO": "valor_efecto",
}

BVG_NUMERIC_COLUMNS = [
    "precio",
    "numero_acciones",
    "valor_efecto",
]

BVG_FALLBACK_MAX_DAYS = 4

REQUIRED_RAW_COLUMNS = {
    "fecha",
    "empresa",
    "close_last",
    "close_vwap",
    "volume_shares_day",
    "turnover_value_day",
    "n_trades_day",
}

HORIZONS = [5]
TARGET_H = "h5"
TEST_SIZE = 30

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
