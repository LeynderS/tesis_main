# Subfase X: Despliegue, trazabilidad y paper trading

Esta subfase no posee un notebook dedicado. La implementación está integrada en el módulo web de Streamlit, porque su objetivo es ejecutar inferencia interactiva, registrar trazabilidad operativa y simular el seguimiento tipo paper trading.

Componentes principales:

- `app/main.py`: orquesta la interfaz, la carga de datos, la inferencia y el flujo de actualización.
- `app/inference.py`: carga artefactos clásicos y cuánticos, reconstruye los insumos necesarios y ejecuta predicciones.
- `app/logs.py`: mantiene el registro de trazabilidad, resuelve predicciones pendientes y ejecuta el ciclo de seguimiento viernes a viernes.

La subfase conserva las rutas centralizadas del proyecto (`DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`) y no redefine lógica compartida fuera de `bvg_core`.
