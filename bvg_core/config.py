from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

DATA_PATH = PROCESSED_DIR / "BVG_features_svc_master.csv"
LOG_PATH = ROOT / "app" / "logs_trazabilidad.csv"
LAST_RUN_PATH = ROOT / "app" / "last_run.csv"

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

REQUIRED_RAW_COLUMNS = {
    "fecha",
    "empresa",
    "close_last",
    "close_vwap",
    "volume_shares_day",
    "turnover_value_day",
    "n_trades_day",
}

COMPANIES = [
    "BANCO GUAYAQUIL S.A.",
    "CORPORACION FAVORITA C.A.",
]

COMPANY_FILE_MAP = {
    "BANCO GUAYAQUIL S.A.": "BANCO_GUAYAQUIL_SA",
    "CORPORACION FAVORITA C.A.": "CORPORACION_FAVORITA_CA",
}

HORIZONS = [5]
GLOBAL_SEED = 42
FEATURE_EXCLUDE = {
    "fecha",
    "empresa",
    "horizonte",
    "target_up_h1",
    "target_up_h5",
    "target_up_h20",
    "ret_fwd_h1",
    "ret_fwd_h5",
    "ret_fwd_h20",
}

TARGET_H = "h5"
TEST_SIZE = 30
