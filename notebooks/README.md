# Notebooks - ETSformer

Este directorio contiene notebooks de análisis y exploración de los experimentos realizados con ETSformer.

## Contenido

### 01_metrics_analysis.ipynb
Análisis exhaustivo de métricas de los experimentos realizados con ETSformer sobre los datasets ETT (Electricity Transformer Temperature).

**Incluye:**
- Carga y visualización de resultados experimentales
- Tablas de MSE/MAE por dataset y horizonte
- Curvas de entrenamiento y learning rate schedules
- Visualización de predicciones vs valores reales
- Experimentos de ablación del parámetro K (Top-K Fourier bases)
- Análisis estadístico y correlaciones
- Exportación de resultados a CSV

---

## Implementación del Modelo

### Arquitectura
ETSformer es un modelo basado en Transformer que descompone series temporales en tres componentes:
- **Level**: Tendencia base de la serie
- **Growth**: Cambios en la tendencia a lo largo del tiempo
- **Seasonality**: Patrones estacionales capturados mediante Frequency Attention

**Características clave:**
- **Exponential Smoothing Attention**: Mecanismo de atención inspirado en suavizado exponencial para modelar nivel y crecimiento
- **Frequency Attention**: Selección de Top-K componentes de Fourier para capturar patrones estacionales
- **Arquitectura modular**: Encoder-Decoder con bloques de atención especializados

### Datasets Evaluados
- **ETTh1, ETTh2**: Datos horarios de temperatura de transformadores eléctricos
- **ETTm1, ETTm2**: Datos minutales (mayor resolución temporal)
- **Variable objetivo**: OT (Oil Temperature)
- **Split**: 70% train, 10% validación, 20% test

---

## Entrenamiento

### Scripts de Ejecución

#### 1. `launch-khipu.sh`
Script principal para ejecutar experimentos completos en todos los datasets y horizontes.

**Configuración:**
```bash
# Datasets
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2")

# Horizontes de predicción
HORIZONS=(24 48 96 192 336 720)

# Parámetros del modelo
SEQ_LEN=96          # Ventana de entrada
D_MODEL=512         # Dimensión del modelo
E_LAYERS=2          # Capas del encoder
D_LAYERS=1          # Capas del decoder
```

**Uso:**
```bash
bash launch-khipu.sh
```

**Output:**
- Resultados guardados en `results/YYYYMMDD_HHMMSS_ETSformer_{dataset}_{horizon}/`
- Checkpoints del mejor modelo (`checkpoint.pth`)
- Predicciones (`pred.npy`, `true.npy`)
- Logs de entrenamiento (`training_log.json`, `test_results.json`)

#### 2. `launch-ablation.sh`
Script para experimentos de ablación del parámetro K (número de frecuencias de Fourier).

**Configuración:**
```bash
# Dataset fijo para ablación
DATASET="ETTh1"

# Horizontes seleccionados
HORIZONS=(24 96)

# Valores de K a evaluar
K_VALUES=(1 3 5 7)
```

**Uso:**
```bash
bash launch-ablation.sh
```

**Output:**
- Resultados guardados en `results_abl/YYYYMMDD_HHMMSS_ETSformer_{dataset}_{horizon}/`
- Permite analizar el impacto de K en el rendimiento del modelo

### Parámetros de Entrenamiento

**Optimización:**
- Optimizer: Adam
- Learning rate: 1e-4 → exponential decay
- Early stopping: patience=3 (basado en validation loss)
- Batch size: 32

**Regularización:**
- Dropout: 0.05
- Data augmentation: Transform con σ=0.5

**Hardware:**
- GPU: NVIDIA (automático si disponible)
- Checkpointing automático del mejor modelo

---

## Evaluación

### Métricas Principales
- **MSE (Mean Squared Error)**: Penaliza errores grandes
- **MAE (Mean Absolute Error)**: Error promedio absoluto
- **RMSE (Root Mean Squared Error)**: Interpretabilidad en la misma escala que los datos
- **MAPE (Mean Absolute Percentage Error)**: Error relativo
- **MSPE (Mean Squared Percentage Error)**: Variante cuadrática del error relativo

### Resultados Principales

**Mejor rendimiento:**
- Dataset: ETTm2
- Horizonte: 24
- MSE: 0.1132 | MAE: 0.2249

**Tendencias observadas:**
1. Correlación fuerte entre horizonte y error (r > 0.78)
2. ETTm2 consistentemente supera a otros datasets
3. Horizontes cortos (h=24) son más confiables que horizontes largos (h=720)

### Análisis de Ablación

**Impacto del parámetro K:**
- K=1: Captura solo la frecuencia dominante
- K=3: Baseline (balance entre complejidad y generalización)
- K=5-7: Mayor capacidad de captura de patrones estacionales

**Hallazgo:** K óptimo varía según horizonte (K=1 para h=24, K=7 para h=96 en ETTh1)

---

## Estructura de Resultados

```
results/
├── YYYYMMDD_HHMMSS_ETSformer_{dataset}_{horizon}/
│   ├── checkpoint.pth          # Mejor modelo
│   ├── pred.npy                # Predicciones
│   ├── true.npy                # Valores reales
│   ├── training_log.json       # Historial de entrenamiento
│   └── test_results.json       # Métricas finales

results_abl/
└── (misma estructura, experimentos de ablación)

notebooks/outputs/
├── all_results.csv             # Todas las métricas
├── mse_by_horizon.csv          # MSE por dataset/horizonte
├── mae_by_horizon.csv          # MAE por dataset/horizonte
└── training_history.csv        # Curvas de entrenamiento
```

---

## Dependencias

Ver `requirements.txt` en el directorio raíz del proyecto.

**Principales:**
- PyTorch >= 1.9
- NumPy, Pandas
- Matplotlib, Seaborn (visualización)
- Scikit-learn (métricas)

---

## Referencias

- Paper original: [ETSformer: Exponential Smoothing Transformers for Time-series Forecasting](https://arxiv.org/abs/2202.01381)
- Implementación base: https://github.com/salesforce/ETSformer
