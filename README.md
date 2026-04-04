# Tesis SVR vs QSVR (BVG)

Proyecto experimental para comparar modelos clasicos y cuanticos en series temporales financieras de la Bolsa de Valores de Guayaquil (BVG).

## Requisitos

- Python 3.12+
- Git

## Inicializar entorno virtual

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (CMD)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS (zsh/bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Uso rapido

Con el entorno activo, abre los notebooks de cada fase y ejecuta en orden:

1. fase1_carga_entendimiento_bvg.ipynb
2. fase2_estructura_integridad_temporal_bvg.ipynb
3. fase3_eda_profundo_retornos_bvg.ipynb
4. fase4_feature_engineering_bvg.ipynb
5. fase5_svr_walkforward_bvg.ipynb
6. fase5_v2_svr_tuning_ffill_comparativo.ipynb
