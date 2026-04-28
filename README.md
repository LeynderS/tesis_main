# Tesis SVR vs QSVR (BVG)

Proyecto de tesis para predecir la dirección (**Sube/Baja**) de acciones de la Bolsa de Valores de Guayaquil (BVG), comparando **SVC clásico** y **QSVC / Kernel cuántico** en un horizonte ganador **h=5 (viernes a viernes)**.

La implementación está organizada por **fases académicas**, con notebooks autocontenidos por subfase.

## Requisitos

- Python 3.12+
- Git

## Estructura por fases

- **Fase 1 — Preparación de datos**
  - Carpeta: `Fase1_Preparacion/`
  - Objetivo: limpieza, consolidación y preprocesamiento inicial.

- **Fase 2 — Características**
  - Carpeta: `Fase2_Caracteristicas/`
  - Objetivo: ingeniería de features y datasets derivados.

- **Fase 3 — Modelado clásico**
  - Carpeta: `Fase3_ModeladoClasico/`
  - Subfase 6 (final h5):
    - Notebook: `subfase_6_modelado_final_h5.ipynb`
    - Modelo: `RobustScaler + SVC(probability=True)`
    - Parámetros fijos:
      - **Banco Guayaquil**: `kernel='linear', C=0.1, class_weight='balanced'`
      - **Corporación Favorita**: `kernel='rbf', C=10, gamma=0.01, class_weight='balanced'`

- **Fase 4 — Modelado cuántico**
  - Carpeta: `Fase4_ModeladoCuantico/`
  - Subfase 7 (h5):
    - Notebook: `subfase_7_modelado_cuantico_h5.ipynb`
    - Flujo: `PCA (≤5) + ZZFeatureMap + FidelityQuantumKernel + SVC (kernel precomputed)`
    - Backend: simulación **statevector**

## Artefactos generados

- **Modelos clásicos**: `models/classical/`
  - `{empresa}_h5_pipeline.joblib`
  - `{empresa}_h5_manifest.json`
  - `h5_subfase6_run_summary.json`

- **Modelos cuánticos**: `models/quantum/`
  - `{empresa}_h5_scaler.joblib`
  - `{empresa}_h5_pca.joblib`
  - `{empresa}_h5_svc.joblib`
  - `{empresa}_h5_kernel_config.json`
  - `{empresa}_h5_manifest.json`
  - `h5_subfase7_run_summary.json`

## Ejecución de subfases (notebook-only)

1. **Subfase 6 — Modelado clásico h5**
   - Abrir: `Fase3_ModeladoClasico/subfase_6_modelado_final_h5.ipynb`
   - Ejecutar todas las celdas

2. **Subfase 7 — Modelado cuántico h5**
   - Abrir: `Fase4_ModeladoCuantico/subfase_7_modelado_cuantico_h5.ipynb`
   - Ejecutar todas las celdas (puede tardar horas por kernel cuántico)

## Documentación de subfases

- `docs/subfase_6_modelado_clasico_h5.md`
- `docs/subfase_7_modelado_cuantico_h5.md`

## Resultados esperados y tiempos de ejecución

> **Nota**: Los resultados pueden variar según la máquina. En especial, el kernel cuántico puede tardar **horas** en ejecutarse por su complejidad cuadrática.

### Comparación de métricas (última ejecución)

Las métricas se leen desde:

- `models/classical/h5_subfase6_run_summary.json`
- `models/quantum/h5_subfase7_run_summary.json`

| Empresa              | Modelo                 | Accuracy |     F1 | Positive rate (pred) | Positive rate (test) |
| -------------------- | ---------------------- | -------: | -----: | -------------------: | -------------------: |
| Banco Guayaquil      | Clásico (SVC)          |   0.8333 | 0.9091 |               1.0000 |               0.8333 |
| Banco Guayaquil      | Cuántico (QKernel+SVC) |   0.3667 | 0.4571 |               0.3333 |               0.8333 |
| Corporación Favorita | Clásico (SVC)          |   0.4333 | 0.5143 |               0.5333 |               0.6333 |
| Corporación Favorita | Cuántico (QKernel+SVC) |   0.4000 | 0.3077 |               0.2333 |               0.6333 |

En esta corrida, el **modelo clásico** supera al cuántico en **accuracy** y **F1** para ambas empresas. Esto sugiere que, bajo la configuración actual y el tamaño de muestra utilizado, el enfoque clásico ofrece mejor rendimiento predictivo; el cuántico queda como línea base comparativa y su desempeño puede estar limitado por la complejidad del kernel y la reducción de dimensionalidad.

### Tiempos de ejecución (referencia)

- Subfase 6 (clásico): minutos.
- Subfase 7 (cuántico): puede tomar **horas** según CPU/RAM.

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
