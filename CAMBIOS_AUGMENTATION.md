# Cambios: Soporte Multi-Método para Data Augmentation

## 🎯 Objetivo

Añadir flexibilidad al notebook para permitir seleccionar entre dos métodos de data augmentation:
- **EDA** (Easy Data Augmentation): Rápido y eficiente para CPU
- **Back-Translation**: Más preciso pero requiere más recursos

## 📝 Cambios Realizados

### 1. Nueva Celda de Configuración (Sección 1.5)

Añadida celda de configuración centralizada que permite:
- Seleccionar método: `AUGMENTATION_METHOD = 'eda'` o `'backtranslation'`
- Configurar parámetros de EDA
- Configurar idiomas y device para back-translation
- Definir estrategia de balanceo (`TARGET_SAMPLES`)

```python
AUGMENTATION_METHOD = 'eda'  # o 'backtranslation'

EDA_CONFIG = {
    'alpha_sr': 0.15,
    'alpha_ri': 0.15,
    'alpha_rs': 0.15,
    'p_rd': 0.1
}

BACKTRANSLATION_CONFIG = {
    'languages': ['en', 'fr', 'de'],
    'device': 'cpu'
}

TARGET_SAMPLES = {
    0: 30000, 1: 30000, 2: 40000, 3: 40000, 4: 50000
}
```

### 2. Reorganización de Funciones (Sección 2)

**Antes:** Todo el código EDA estaba en una sola celda gigante

**Ahora:** Separado en subsecciones:
- **Sección 2.1**: Funciones EDA (STOP_WORDS, SYNONYMS, eda(), synonym_replacement(), etc.)
- **Sección 2.2**: Funciones Back-Translation (load_backtranslation_models(), translate_text(), back_translate())

### 3. Función Unificada de Augmentation (Sección 3)

Nueva función `augment_class()` que:
- Recibe parámetro `method` ('eda' o 'backtranslation')
- Selecciona automáticamente el método según configuración
- Maneja la carga de modelos de traducción (lazy loading)
- Usa configuraciones de `EDA_CONFIG` o `BACKTRANSLATION_CONFIG`

```python
def augment_class(df, polarity, target_size, method='eda'):
    """Aumenta una clase usando el método especificado"""
    if method == 'eda':
        # Usa funciones EDA
        aug_texts = eda(row['Review'], **EDA_CONFIG, num_aug=1)
    elif method == 'backtranslation':
        # Carga modelos (solo primera vez) y traduce
        aug_texts = back_translate(row['Review'], ...)
```

### 4. Undersampling Actualizado (Sección 4)

- Ahora usa `TARGET_SAMPLES` de la configuración
- Código más limpio y mantenible

### 5. Renumeración de Secciones

| Antes | Ahora | Contenido |
|-------|-------|-----------|
| - | 1.5 | Configuración de Data Augmentation |
| 2 | 2 | Funciones de Data Augmentation |
| - | 2.1 | Funciones EDA |
| - | 2.2 | Funciones Back-Translation |
| 3 | 3 | Aplicar Data Augmentation |
| 4 | 4 | Undersampling |
| 5 | 5 | Generación de Embeddings |
| 6 | 6 | Exportar Embeddings a CSV |
| 7 | 7 | Split Train/Test |
| 8 | 8 | Modelo Polaridad |
| 9 | 9 | Evaluación Polaridad |
| 10 | 10 | Modelo Tipo |
| 11 | 11 | Evaluación Tipo |
| 12 | 12 | Guardar Modelos y Embeddings |

### 6. Metadata Actualizada

El archivo `models_metadata.json` ahora incluye:
```json
{
  "augmentation_method": "eda",
  "eda_config": {...},  // o "backtranslation_config": {...}
  "target_samples": {...},
  ...
}
```

### 7. Requirements Actualizados

Añadido comentario en `requirements.txt`:
```
# Back-Translation (opcional - solo si usas AUGMENTATION_METHOD='backtranslation')
# torch>=2.0.0
# transformers>=4.21.0
# sacremoses>=0.0.53
```

### 8. Documentación

Nuevos archivos:
- `README_AUGMENTATION.md`: Guía completa con ejemplos, comparaciones y troubleshooting

## 🚀 Uso

### Opción 1: EDA (Recomendado para CPU, rápido)

1. En celda 1.5, configurar:
```python
AUGMENTATION_METHOD = 'eda'
```

2. Ejecutar notebook normalmente
3. Tiempo estimado: ~15-20 minutos total

### Opción 2: Back-Translation (GPU recomendado, más preciso)

1. Instalar dependencias:
```bash
pip install torch transformers sacremoses
```

2. En celda 1.5, configurar:
```python
AUGMENTATION_METHOD = 'backtranslation'
BACKTRANSLATION_CONFIG = {
    'languages': ['en'],  # Solo inglés para velocidad
    'device': 'cuda'  # o 'cpu' si no hay GPU
}
```

3. Ejecutar notebook
4. Tiempo estimado: ~2-4 horas en CPU, ~30-60 min en GPU

## 📊 Comparación

| Aspecto | EDA | Back-Translation |
|---------|-----|------------------|
| **Tiempo (50K samples)** | ~5-10 min | ~2-4 horas (CPU) |
| **Memoria** | ~2 GB | ~8-12 GB |
| **GPU** | No necesaria | Recomendada |
| **Dependencias** | Mínimas | torch, transformers |
| **Calidad** | Buena | Excelente |

## 🎓 Ventajas del Nuevo Diseño

1. **Flexibilidad**: Cambiar método editando 1 línea
2. **Mantenibilidad**: Código organizado en funciones separadas
3. **Escalabilidad**: Fácil añadir nuevos métodos
4. **Documentación**: Metadata incluye método usado
5. **Performance**: Lazy loading de modelos (solo carga si necesario)

## ⚙️ Arquitectura

```
Configuración (1.5)
    ↓
Funciones (2.1, 2.2)
    ↓
augment_class() ← Selecciona método
    ↓
Dataset Balanceado (3, 4)
    ↓
Embeddings (5)
    ↓
Modelos (6-11)
    ↓
Export (12)
```

## 🔄 Migración desde Versión Anterior

Si estás usando la versión anterior del notebook:

1. **No necesitas cambiar nada** para usar EDA (comportamiento por defecto)
2. Para cambiar a back-translation:
   - Instalar dependencias
   - Cambiar `AUGMENTATION_METHOD = 'backtranslation'`
3. Tus archivos de salida seguirán siendo compatibles con Scala

## 📦 Archivos Afectados

- ✅ `02_model_pipeline.ipynb` - Restructurado con soporte multi-método
- ✅ `requirements.txt` - Añadidas dependencias opcionales comentadas
- ✅ `README_AUGMENTATION.md` - Nueva guía completa
- ✅ `CAMBIOS_AUGMENTATION.md` - Este archivo

## ✅ Testing

Ambos métodos fueron diseñados para producir el mismo formato de salida:
- `embeddings_complete.csv`
- `embeddings_with_split.csv`
- `model_polarity.pkl`
- `model_type.pkl`
- `models_metadata.json`

La única diferencia es la calidad/diversidad del texto augmentado.

## 🐛 Troubleshooting

Ver `README_AUGMENTATION.md` sección "Troubleshooting" para soluciones a problemas comunes.
