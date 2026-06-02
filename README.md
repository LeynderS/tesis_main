# Tesis SVC vs QSVC (BVG)

Proyecto de tesis para predecir la dirección (**Sube/Baja**) de acciones de la Bolsa de Valores de Guayaquil (BVG), comparando **SVC clásico** y **QSVC / Kernel cuántico** en el horizonte ganador **h=5**.

La implementación está organizada en **5 fases metodológicas y 11 subfases**, con notebooks autocontenidos como evidencia técnica y un módulo web en Streamlit para inferencia, trazabilidad y seguimiento experimental.

## Requisitos

- Python 3.12+
- Git
- Dependencias de `requirements.txt`
- Instalación editable del paquete local con `pip install -e .`

## Inicializar entorno virtual

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Windows CMD

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Estructura metodológica

| Fase                                    | Carpeta                         | Subfase                           | Evidencia                                        |
| --------------------------------------- | ------------------------------- | --------------------------------- | ------------------------------------------------ |
| I. Preparación y Preprocesamiento       | `Fase1_Preparacion/`            | I. Ingesta, limpieza y control    | `subfase_1_ingesta_limpieza_control.ipynb`       |
| I. Preparación y Preprocesamiento       | `Fase1_Preparacion/`            | II. Evaluación de integridad      | `subfase_2_evaluacion_integridad.ipynb`          |
| II. Ingeniería de Características       | `Fase2_Caracteristicas/`        | III. EDA multihorizonte           | `subfase_3_eda_multihorizonte.ipynb`             |
| II. Ingeniería de Características       | `Fase2_Caracteristicas/`        | IV. Feature engineering SVC       | `subfase_4_feature_engineering_svc.ipynb`        |
| III. Modelado Clásico y Selección       | `Fase3_ModeladoClasico/`        | V. Exploración y optimización SVC | `subfase_5_exploracion_y_optimizacion_svc.ipynb` |
| III. Modelado Clásico y Selección       | `Fase3_ModeladoClasico/`        | VI. Entrenamiento final SVC       | `subfase_6_entrenamiento_final_svc.ipynb`        |
| IV. Implementación Cuántica y Contraste | `Fase4_ModeladoCuantico/`       | VII. Entrenamiento QSVC           | `subfase_7_entrenamiento_qsvc.ipynb`             |
| IV. Implementación Cuántica y Contraste | `Fase4_ModeladoCuantico/`       | VIII. Validación PCA              | `subfase_8_validacion_pca.ipynb`                 |
| IV. Implementación Cuántica y Contraste | `Fase4_ModeladoCuantico/`       | IX. Calibración y contraste       | `subfase_9_calibracion_y_contraste.ipynb`        |
| V. Validación Experimental y Despliegue | `Fase5_ValidacionExperimental/` | X. Despliegue y trazabilidad      | `subfase_10_despliegue_trazabilidad.md`          |
| V. Validación Experimental y Despliegue | `Fase5_ValidacionExperimental/` | XI. Reentrenamiento adaptativo    | `subfase_11_reentrenamiento_adaptativo.ipynb`    |

## Flujo de datos

```text
data/raw/BVG_Acciones.csv
  -> data/processed/BVG_Acciones_limpio.csv
  -> data/processed/BVG_features_svc_master.csv
  -> models/classical/
  -> models/quantum/
  -> results/fase4_comparativa/
  -> app/ Streamlit
```

## Artefactos principales

**Datos procesados**

- `data/processed/BVG_Acciones_limpio.csv`
- `data/processed/BVG_features_svc_master.csv`
- `data/processed/BVG_features_svc_dictionary.csv`

**Modelos clásicos**

- `models/classical/{empresa}_h5_pipeline.joblib`
- `models/classical/{empresa}_h5_manifest.json`
- `models/classical/h5_subfase6_run_summary.json`

**Modelos cuánticos**

- `models/quantum/{empresa}_h5_scaler.joblib`
- `models/quantum/{empresa}_h5_pca.joblib`
- `models/quantum/{empresa}_h5_svc.joblib`
- `models/quantum/{empresa}_h5_kernel_config.json`
- `models/quantum/{empresa}_h5_manifest.json`
- `models/quantum/h5_subfase7_run_summary.json`

**Resultados comparativos**

- `results/fase3_modelado_clasico/subfase5/`
- `results/fase4_comparativa/BVG_pca_varianza_explicada.csv`
- `results/fase4_comparativa/BVG_subfase8b_calibracion.csv`
- `results/fase4_comparativa/BVG_subfase8_test_estadistico.csv`
- `results/fase4_comparativa/BVG_subfase8_comparativa_h5.csv`

## Métricas finales h=5

Las métricas se leen desde los resúmenes de ejecución y manifiestos generados en `models/classical/`, `models/quantum/` y `results/fase4_comparativa/`.

| Empresa              | Modelo                | Accuracy |     F1 | Positive rate pred | Positive rate test |
| -------------------- | --------------------- | -------: | -----: | -----------------: | -----------------: |
| Banco Guayaquil      | Clásico SVC           |   0.8333 | 0.9091 |             1.0000 |             0.8333 |
| Banco Guayaquil      | QSVC / Quantum Kernel |   0.3667 | 0.4571 |             0.3333 |             0.8333 |
| Corporación Favorita | Clásico SVC           |   0.4333 | 0.5143 |             0.5333 |             0.6333 |
| Corporación Favorita | QSVC / Quantum Kernel |   0.4000 | 0.3077 |             0.2333 |             0.6333 |

En la ejecución registrada, el modelo clásico supera al cuántico en accuracy y F1 para ambas empresas. El contraste estadístico de Fase IV reporta diferencia significativa para Banco Guayaquil y no significativa para Corporación Favorita.

## Aplicación web

El módulo `app/` implementa la subfase de despliegue y trazabilidad:

- `app/main.py`: orquestación de Streamlit.
- `app/inference.py`: carga de artefactos y predicción clásica/cuántica.
- `app/logs.py`: bitácora de trazabilidad y resolución de predicciones pendientes.
- `app/retrain.py`: reentrenamiento clásico versionado.

Para ejecutar la aplicación:

```powershell
streamlit run app/main.py
```

## Notas de reproducibilidad

- Las rutas compartidas y constantes del proyecto viven en `bvg_core/config.py`.
- Los notebooks usan `bvg_core` para lógica reutilizable y mantienen el flujo por subfase.
- El reentrenamiento no sobrescribe artefactos originales; genera salidas versionadas en carpetas `retrained/`.
- El flujo cuántico puede tomar más tiempo que el clásico por la construcción de matrices de kernel.
