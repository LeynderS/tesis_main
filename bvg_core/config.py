from __future__ import annotations

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

TARGET_H = "h5"
TEST_SIZE = 30
