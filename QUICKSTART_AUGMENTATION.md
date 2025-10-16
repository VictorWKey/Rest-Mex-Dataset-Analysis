# 🚀 Inicio Rápido - Data Augmentation

## Configuración en 3 Pasos

### 1️⃣ Selecciona tu método

Edita la celda **1.5** del notebook:

```python
# Para desarrollo rápido (recomendado):
AUGMENTATION_METHOD = 'eda'

# Para máxima calidad (lento):
AUGMENTATION_METHOD = 'backtranslation'
```

### 2️⃣ Instala dependencias (solo si usas back-translation)

```bash
# Solo si elegiste AUGMENTATION_METHOD = 'backtranslation'
pip install torch transformers sacremoses
```

### 3️⃣ Ejecuta el notebook

```bash
# Instalar dependencias base
pip install -r requirements.txt

# Ejecutar Jupyter
jupyter notebook 02_model_pipeline.ipynb
```

---

## ⚡ Comparación Rápida

| Método | Tiempo | GPU | Calidad | Recomendado para |
|--------|--------|-----|---------|------------------|
| **EDA** | 10 min | ❌ No | ⭐⭐⭐⭐ | Desarrollo rápido, CPU |
| **Back-Translation** | 2-4 hrs | ✅ Sí | ⭐⭐⭐⭐⭐ | Producción, GPU disponible |

---

## 🎯 Casos de Uso

### Caso 1: "Tengo 2-4 horas y CPU"
```python
AUGMENTATION_METHOD = 'eda'
```
✅ Perfecto. EDA es rápido y eficiente en CPU.

### Caso 2: "Tengo GPU y quiero máxima calidad"
```python
AUGMENTATION_METHOD = 'backtranslation'
BACKTRANSLATION_CONFIG = {'languages': ['en', 'fr', 'de'], 'device': 'cuda'}
```
✅ Back-translation con múltiples idiomas.

### Caso 3: "Tengo CPU pero puedo esperar 4+ horas"
```python
AUGMENTATION_METHOD = 'backtranslation'
BACKTRANSLATION_CONFIG = {'languages': ['en'], 'device': 'cpu'}
```
⚠️ Solo usa inglés para reducir tiempo.

---

## 📂 Archivos Generados (ambos métodos)

Ambos métodos generan los mismos archivos de salida:

```
embeddings_complete.csv       ← Todos los embeddings + labels
embeddings_with_split.csv      ← Con columna train/test
model_polarity.pkl             ← Modelo de polaridad (sklearn)
model_type.pkl                 ← Modelo de tipo (sklearn)
models_metadata.json           ← Metadata (incluye método usado)
```

Todos compatibles con Scala/Spark MLlib.

---

## 🐛 Problemas Comunes

### "ModuleNotFoundError: No module named 'transformers'"
```bash
pip install torch transformers sacremoses
```

### "Back-translation muy lento en CPU"
Cambia a:
```python
AUGMENTATION_METHOD = 'eda'
```

### "CUDA out of memory"
```python
BACKTRANSLATION_CONFIG = {'languages': ['en'], 'device': 'cpu'}
```

---

## 📚 Documentación Completa

Para más detalles, ver:
- `README_AUGMENTATION.md` - Guía completa con ejemplos
- `CAMBIOS_AUGMENTATION.md` - Cambios técnicos implementados
- `README_SCALA.md` - Cómo usar los archivos generados en Scala

---

## ✅ Checklist Pre-Ejecución

- [ ] Elegir método en celda 1.5: `AUGMENTATION_METHOD = 'eda'` o `'backtranslation'`
- [ ] Si usas back-translation: `pip install torch transformers sacremoses`
- [ ] Instalar: `pip install -r requirements.txt`
- [ ] Verificar tiempo disponible: EDA ~10-15 min, Back-translation ~2-4 hrs
- [ ] Ejecutar notebook completo

---

## 💡 Recomendación

**Para la mayoría de casos, usa EDA:**
- ✅ 10x más rápido
- ✅ No requiere GPU
- ✅ Resultados muy buenos
- ✅ Ideal para desarrollo iterativo

**Usa back-translation solo si:**
- Tienes GPU disponible
- Necesitas máxima calidad
- Puedes esperar 2-4 horas
- Es para modelo de producción final
