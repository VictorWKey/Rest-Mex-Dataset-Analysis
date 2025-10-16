# Guía de Data Augmentation

Este notebook soporta dos métodos de data augmentation para balancear las clases minoritarias:

## 📋 Métodos Disponibles

### 1. EDA (Easy Data Augmentation) ⚡ RECOMENDADO

**Ventajas:**
- ✅ Muy rápido (~5-10 minutos para 50K+ muestras)
- ✅ No requiere GPU
- ✅ Bajo uso de memoria
- ✅ Resultados efectivos para español

**Desventajas:**
- ⚠️ Menor diversidad que back-translation
- ⚠️ Requiere diccionario de sinónimos

**Técnicas aplicadas:**
- **Synonym Replacement**: Reemplaza palabras con sinónimos (15% de palabras)
- **Random Insertion**: Inserta palabras aleatorias (15% de palabras)
- **Random Swap**: Intercambia posiciones de palabras (15% de palabras)
- **Random Deletion**: Elimina palabras aleatoriamente (10% probabilidad)

**Uso:**
```python
AUGMENTATION_METHOD = 'eda'  # En la celda de configuración
```

---

### 2. Back-Translation 🌍 (Más preciso pero lento)

**Ventajas:**
- ✅ Mayor diversidad semántica
- ✅ Preserva mejor el significado original
- ✅ Soporta múltiples idiomas intermedios

**Desventajas:**
- ⚠️ Muy lento (~2-4 horas en CPU para 50K+ muestras)
- ⚠️ Requiere modelos de traducción pesados (~2-3 GB de memoria)
- ⚠️ Necesita dependencias adicionales (torch, transformers)

**Proceso:**
1. Español → Inglés/Francés/Alemán
2. Idioma intermedio → Español
3. Resultado: Variación semántica del texto original

**Uso:**
```python
AUGMENTATION_METHOD = 'backtranslation'  # En la celda de configuración
```

**Dependencias adicionales:**
Descomenta en `requirements.txt`:
```bash
torch>=2.0.0
transformers>=4.21.0
sacremoses>=0.0.53
```

Luego instala:
```bash
pip install torch transformers sacremoses
```

---

## ⚙️ Configuración

### Configuración de EDA

```python
EDA_CONFIG = {
    'alpha_sr': 0.15,  # Porcentaje para synonym replacement
    'alpha_ri': 0.15,  # Porcentaje para random insertion
    'alpha_rs': 0.15,  # Porcentaje para random swap
    'p_rd': 0.1        # Probabilidad de random deletion
}
```

**Recomendaciones:**
- Para textos cortos (<30 palabras): Reducir `alpha_*` a 0.10
- Para textos largos (>100 palabras): Aumentar `alpha_*` a 0.20
- `p_rd` siempre mantener bajo (0.05-0.15) para no perder información

### Configuración de Back-Translation

```python
BACKTRANSLATION_CONFIG = {
    'languages': ['en', 'fr', 'de'],  # Idiomas intermedios
    'device': 'cpu'  # 'cuda' si tienes GPU disponible
}
```

**Idiomas soportados:**
- `'en'` - Inglés (más rápido, mejor calidad)
- `'fr'` - Francés
- `'de'` - Alemán
- `'it'` - Italiano
- `'pt'` - Portugués

**Nota:** Cada idioma adicional aumenta el tiempo de ejecución. Para CPU, se recomienda usar solo `['en']`.

### Estrategia de Balanceo

```python
TARGET_SAMPLES = {
    0: 30000,  # Pol 1: 5K → 30K (6x oversampling)
    1: 30000,  # Pol 2: 5K → 30K (6x oversampling)
    2: 40000,  # Pol 3: 15K → 40K (2.7x oversampling)
    3: 40000,  # Pol 4: 45K → 40K (undersampling)
    4: 50000   # Pol 5: 136K → 50K (undersampling)
}
```

**Personalización:**
- Ajusta los valores según tu dataset
- Aumenta `TARGET_SAMPLES[0]` y `TARGET_SAMPLES[1]` si tienes muy pocas muestras en clases 1-2
- Reduce `TARGET_SAMPLES[4]` si quieres dataset más pequeño (más rápido)

---

## 📊 Comparación de Rendimiento

| Característica | EDA | Back-Translation |
|---------------|-----|------------------|
| **Tiempo (50K muestras)** | 5-10 min | 2-4 horas |
| **Memoria RAM** | ~2 GB | ~8-12 GB |
| **GPU necesaria** | No | Recomendado |
| **Calidad semántica** | Buena | Excelente |
| **Diversidad** | Media | Alta |
| **Complejidad** | Baja | Alta |

---

## 🚀 Recomendaciones de Uso

### Caso 1: Desarrollo rápido (2-4 horas disponibles)
```python
AUGMENTATION_METHOD = 'eda'
EDA_CONFIG = {'alpha_sr': 0.15, 'alpha_ri': 0.15, 'alpha_rs': 0.15, 'p_rd': 0.1}
```

### Caso 2: Máxima calidad (8+ horas disponibles, GPU disponible)
```python
AUGMENTATION_METHOD = 'backtranslation'
BACKTRANSLATION_CONFIG = {'languages': ['en', 'fr', 'de'], 'device': 'cuda'}
```

### Caso 3: Equilibrio (4-6 horas, CPU)
```python
AUGMENTATION_METHOD = 'backtranslation'
BACKTRANSLATION_CONFIG = {'languages': ['en'], 'device': 'cpu'}
```

### Caso 4: Dataset pequeño (<10K muestras)
```python
AUGMENTATION_METHOD = 'backtranslation'
BACKTRANSLATION_CONFIG = {'languages': ['en'], 'device': 'cpu'}
TARGET_SAMPLES = {0: 5000, 1: 5000, 2: 8000, 3: 8000, 4: 10000}
```

---

## 📝 Ejemplos de Salida

### EDA Original:
```
"La comida estuvo excelente y el servicio fue rápido"
```

### EDA Augmented:
```
"La comida estuvo genial y el servicio fue veloz"
"La comida rápido estuvo excelente y el servicio fue magnífico"
```

### Back-Translation (ES→EN→ES):
```
"La comida fue excelente y el servicio fue rápido"
```

---

## 🐛 Troubleshooting

### Error: "No module named 'transformers'"
**Solución:** Instala las dependencias de back-translation:
```bash
pip install torch transformers sacremoses
```

### Error: "CUDA out of memory"
**Solución:** Cambia a CPU:
```python
BACKTRANSLATION_CONFIG = {'languages': ['en'], 'device': 'cpu'}
```

### Error: "KeyError en SYNONYMS"
**Solución:** Expande el diccionario `SYNONYMS` en la celda 2.1 con más palabras relevantes a tu dominio.

### Advertencia: "Back-translation muy lento"
**Solución:** 
1. Reduce idiomas: `['en']` en vez de `['en', 'fr', 'de']`
2. Usa EDA en su lugar
3. Reduce `TARGET_SAMPLES`

---

## 📚 Referencias

- **EDA Paper**: [Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)
- **Back-Translation**: [Understanding Back-Translation at Scale](https://arxiv.org/abs/1808.09381)
- **Helsinki NLP Models**: [OPUS-MT Translation Models](https://huggingface.co/Helsinki-NLP)

---

## ✅ Checklist Pre-Ejecución

- [ ] Decidir método: `'eda'` (rápido) o `'backtranslation'` (preciso)
- [ ] Configurar `AUGMENTATION_METHOD` en celda 1.5
- [ ] Ajustar parámetros en `EDA_CONFIG` o `BACKTRANSLATION_CONFIG`
- [ ] Verificar `TARGET_SAMPLES` según necesidades
- [ ] Si usas back-translation: Instalar torch/transformers
- [ ] Estimar tiempo: EDA ~10 min, Back-translation ~2-4 hrs
- [ ] Verificar espacio en disco: ~500 MB para archivos de salida

---

## 💡 Tip Final

Para la mayoría de casos, **EDA es suficiente y mucho más eficiente**. Solo usa back-translation si:
- Tienes GPU disponible
- Necesitas máxima calidad semántica
- Tienes tiempo suficiente (>4 horas)
- El dataset es pequeño (<20K muestras a augmentar)
