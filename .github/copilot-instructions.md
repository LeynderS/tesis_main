# Contexto del proyecto de tesis

Pipeline experimental para clasificación direccional en acciones BVG usando:

- SVM (baseline)
- QSVM (comparación)

---

# Reglas de implementación

- Código modular por fases
- Cada archivo tiene una única responsabilidad
- No mezclar lógica de datos con modelos

---

# Flujo obligatorio

Cada fase debe responder:

1. ¿Qué hice?
2. ¿Por qué lo hice?
3. ¿Qué resultado obtuve?
4. ¿Qué significa ese resultado?

---

# Datos

Fuente:

- Excel BVG (transacciones crudas)

Transformaciones obligatorias:

- Limpieza de encabezados
- Conversión a formato tabular limpio
- Agregación diaria por emisor

---

# Features

Solo usar features que:

- estén respaldadas por literatura
- tengan sentido financiero

---

# Modelado

- SVM como baseline obligatorio
- Comparar kernels
- No optimizar hiperparámetros sin control

---

# Evaluación

Mínimo:

- Accuracy
- F1-score
- Directional accuracy

---

# Backtesting

Debe simular:

- predicción en tiempo real
- evaluación acumulada

---

# Regla final

Si una fase está mal definida:

→ NO se avanza
→ Se corrige
