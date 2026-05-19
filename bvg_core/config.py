from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

DATASET_RAW_PATH = DATA_RAW / "BVG_Acciones.csv"
DATASET_PROCESSED_PATH = PROCESSED_DIR / "BVG_Acciones_limpio.csv"
DATA_MASTER_PATH = PROCESSED_DIR / "BVG_features_svc_master.csv"
DATA_DICT_PATH = PROCESSED_DIR / "BVG_features_svc_dictionary.csv"
DATA_PATH = DATA_MASTER_PATH  # alias for backward compat
CLASSICAL_DIR = MODELS_DIR / "classical"
QUANTUM_DIR = MODELS_DIR / "quantum"
LOG_PATH = ROOT / "app" / "logs_trazabilidad.csv"
LAST_RUN_PATH = ROOT / "app" / "last_run.csv"
FASE4_RESULTS = RESULTS_DIR / "fase4_comparativa"

EMISOR_COL = 'EMISOR'
FECHA_COL = 'FECHA NEGOCIACIÓN'
PRECIO_COL = 'PRECIO'
ACCIONES_COL = 'NÚMERO DE ACCIONES'
VALOR_EFECTO_COL = 'VALOR EFECTO'
VALOR_NOMINAL_COL = 'VALOR NOMINAL'

EMPRESA_COL = 'empresa'
FECHA_COL_2 = 'fecha'
CLOSE_LAST_COL = 'close_last'
CLOSE_VWAP_COL = 'close_vwap'
VOLUME_SHARES_DAY_COL = 'volume_shares_day'
TURNOVER_VALUE_DAY_COL = 'turnover_value_day'
N_TRADES_DAY_COL = 'n_trades_day'
TARGET_COL = 'target_up_h5'

BVG_REQUIRED_COLUMNS = {
    FECHA_COL,
    EMISOR_COL,
    PRECIO_COL,
    ACCIONES_COL,
    VALOR_EFECTO_COL,
}

REQUIRED_RAW_COLUMNS = {
    FECHA_COL_2,
    EMPRESA_COL,
    CLOSE_LAST_COL,
    CLOSE_VWAP_COL,
    VOLUME_SHARES_DAY_COL,
    TURNOVER_VALUE_DAY_COL,
    N_TRADES_DAY_COL,
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
    FECHA_COL_2,
    EMPRESA_COL,
    "target_up_h1",
    "target_up_h5",
    "target_up_h20",
    "ret_fwd_h1",
    "ret_fwd_h5",
    "ret_fwd_h20",
}

TARGET_H = "h5"
TEST_SIZE = 30
