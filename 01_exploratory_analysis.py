"""
Análisis Exploratorio del Dataset Rest-Mex
==========================================
Este script realiza un análisis exhaustivo de las distribuciones de clases
para determinar la mejor estrategia de balanceo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Cargar datos
print("=" * 80)
print("ANÁLISIS EXPLORATORIO - DATASET REST-MEX")
print("=" * 80)

df = pd.read_csv('data/Rest-Mex_2025_train.csv')

print(f"\n📊 INFORMACIÓN GENERAL")
print(f"{'─' * 80}")
print(f"Total de registros: {len(df):,}")
print(f"Columnas: {list(df.columns)}")
print(f"\nTipos de datos:")
print(df.dtypes)
print(f"\nValores nulos:")
print(df.isnull().sum())

# Identificar la columna de texto - Priorizar 'Review' sobre 'Title'
text_columns = [col for col in df.columns if df[col].dtype == 'object' and col not in ['polaridad', 'tipo_atraccion', 'type', 'attraction_type', 'Polarity', 'Type']]

# Buscar específicamente columnas de review/opinion
review_cols = [col for col in text_columns if 'review' in col.lower() or 'opinion' in col.lower() or 'text' in col.lower()]
title_cols = [col for col in text_columns if 'title' in col.lower() or 'titulo' in col.lower()]

if review_cols:
    text_col = review_cols[0]
    print(f"\n📝 Columna de texto (REVIEW) identificada: '{text_col}'")
elif title_cols:
    text_col = title_cols[0]
    print(f"\n⚠️ Solo se encontró columna de TÍTULO: '{text_col}' (no es la reseña completa)")
elif text_columns:
    text_col = text_columns[0]
    print(f"\n📝 Columna de texto identificada: '{text_col}'")
else:
    text_col = None
    print("\n⚠️ No se identificó columna de texto automáticamente")

# ==============================================================================
# 1. ANÁLISIS DE POLARIDAD
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("1️⃣  ANÁLISIS DE POLARIDAD")
print(f"{'=' * 80}")

# Buscar columna de polaridad (puede tener diferentes nombres)
polarity_cols = [col for col in df.columns if 'polari' in col.lower() or 'rating' in col.lower() or 'star' in col.lower()]
if polarity_cols:
    polarity_col = polarity_cols[0]
else:
    # Intentar encontrar por valores únicos (1-5)
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and df[col].nunique() <= 5 and df[col].min() >= 1 and df[col].max() <= 5:
            polarity_col = col
            break

print(f"\nColumna de polaridad: '{polarity_col}'")

polarity_counts = df[polarity_col].value_counts().sort_index()
polarity_pcts = df[polarity_col].value_counts(normalize=True).sort_index() * 100

print(f"\n📈 Distribución de Polaridad:")
print(f"{'─' * 80}")
for rating in sorted(df[polarity_col].unique()):
    count = polarity_counts.get(rating, 0)
    pct = polarity_pcts.get(rating, 0)
    bar = '█' * int(pct / 2)
    print(f"  Polaridad {rating}: {count:>6,} ({pct:>5.2f}%) {bar}")

# Calcular imbalance ratio
max_class = polarity_counts.max()
min_class = polarity_counts.min()
imbalance_ratio = max_class / min_class

print(f"\n⚖️  MÉTRICAS DE DESBALANCEO:")
print(f"  • Clase mayoritaria: {polarity_counts.idxmax()} con {max_class:,} muestras")
print(f"  • Clase minoritaria: {polarity_counts.idxmin()} con {min_class:,} muestras")
print(f"  • Ratio de desbalanceo: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 10:
    print(f"  ⚠️  DESBALANCEO SEVERO (>10:1) - Se recomienda técnicas agresivas")
elif imbalance_ratio > 5:
    print(f"  ⚠️  DESBALANCEO MODERADO (>5:1) - Se recomienda balanceo")
else:
    print(f"  ✓  DESBALANCEO LEVE (<5:1) - Class weights pueden ser suficientes")

# ==============================================================================
# 2. ANÁLISIS DE TIPO DE ATRACCIÓN
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("2️⃣  ANÁLISIS DE TIPO DE ATRACCIÓN")
print(f"{'=' * 80}")

# Buscar columna de tipo
type_cols = [col for col in df.columns if 'tipo' in col.lower() or 'type' in col.lower() or 'atraccion' in col.lower() or 'attraction' in col.lower()]
if type_cols:
    type_col = type_cols[0]
else:
    # Buscar columna categórica con pocos valores únicos
    for col in df.columns:
        if col != polarity_col and df[col].dtype == 'object' and df[col].nunique() <= 5:
            type_col = col
            break

print(f"\nColumna de tipo: '{type_col}'")

type_counts = df[type_col].value_counts()
type_pcts = df[type_col].value_counts(normalize=True) * 100

print(f"\n📈 Distribución de Tipo de Atracción:")
print(f"{'─' * 80}")
for tipo in type_counts.index:
    count = type_counts[tipo]
    pct = type_pcts[tipo]
    bar = '█' * int(pct / 2)
    print(f"  {tipo:20s}: {count:>6,} ({pct:>5.2f}%) {bar}")

# Calcular imbalance ratio
max_type = type_counts.max()
min_type = type_counts.min()
type_imbalance = max_type / min_type

print(f"\n⚖️  MÉTRICAS DE DESBALANCEO:")
print(f"  • Clase mayoritaria: {type_counts.idxmax()} con {max_type:,} muestras")
print(f"  • Clase minoritaria: {type_counts.idxmin()} con {min_type:,} muestras")
print(f"  • Ratio de desbalanceo: {type_imbalance:.2f}:1")

if type_imbalance > 5:
    print(f"  ⚠️  DESBALANCEO SIGNIFICATIVO - Considerar balanceo")
elif type_imbalance > 2:
    print(f"  ⚠️  DESBALANCEO LEVE - Class weights recomendados")
else:
    print(f"  ✓  DISTRIBUCIÓN ACEPTABLE")

# ==============================================================================
# 3. ANÁLISIS CRUZADO
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("3️⃣  ANÁLISIS CRUZADO (Polaridad × Tipo)")
print(f"{'=' * 80}")

cross_tab = pd.crosstab(df[polarity_col], df[type_col], margins=True)
print("\n📊 Tabla de Contingencia (conteos):")
print(cross_tab)

cross_tab_pct = pd.crosstab(df[polarity_col], df[type_col], normalize='all') * 100
print("\n📊 Tabla de Contingencia (porcentajes):")
print(cross_tab_pct.round(2))

# Identificar combinaciones raras
print(f"\n⚠️  COMBINACIONES CON <1% DE LOS DATOS:")
for pol in df[polarity_col].unique():
    for tipo in df[type_col].unique():
        count = len(df[(df[polarity_col] == pol) & (df[type_col] == tipo)])
        pct = (count / len(df)) * 100
        if pct < 1 and count > 0:
            print(f"  • Polaridad {pol} + {tipo}: {count} muestras ({pct:.2f}%)")

# ==============================================================================
# 4. ANÁLISIS DE TEXTO
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("4️⃣  ANÁLISIS DE TEXTO")
print(f"{'=' * 80}")

# Analizar ambas columnas: Title y Review
for col in ['Title', 'Review']:
    if col in df.columns:
        print(f"\n{'─' * 80}")
        print(f"📝 Análisis de columna: '{col}'")
        print(f"{'─' * 80}")
        
        df[f'{col}_length'] = df[col].astype(str).str.len()
        df[f'{col}_words'] = df[col].astype(str).str.split().str.len()
        
        print(f"\n📊 Estadísticas de Longitud:")
        print(f"  Caracteres:")
        print(f"    • Media: {df[f'{col}_length'].mean():.1f}")
        print(f"    • Mediana: {df[f'{col}_length'].median():.1f}")
        print(f"    • Std: {df[f'{col}_length'].std():.1f}")
        print(f"    • Min: {df[f'{col}_length'].min()}")
        print(f"    • Max: {df[f'{col}_length'].max()}")
        print(f"    • Q1 (25%): {df[f'{col}_length'].quantile(0.25):.1f}")
        print(f"    • Q3 (75%): {df[f'{col}_length'].quantile(0.75):.1f}")
        
        print(f"\n  Palabras:")
        print(f"    • Media: {df[f'{col}_words'].mean():.1f}")
        print(f"    • Mediana: {df[f'{col}_words'].median():.1f}")
        print(f"    • Std: {df[f'{col}_words'].std():.1f}")
        print(f"    • Min: {df[f'{col}_words'].min()}")
        print(f"    • Max: {df[f'{col}_words'].max()}")
        print(f"    • Q1 (25%): {df[f'{col}_words'].quantile(0.25):.1f}")
        print(f"    • Q3 (75%): {df[f'{col}_words'].quantile(0.75):.1f}")
        
        print(f"\n📝 Ejemplos de {col} por polaridad:")
        for pol in sorted(df[polarity_col].unique()):
            print(f"\n  Polaridad {pol}:")
            sample = df[df[polarity_col] == pol][col].iloc[0]
            if len(str(sample)) > 200:
                print(f"    {sample[:200]}...")
            else:
                print(f"    {sample}")

# Determinar cuál usar
if 'Review' in df.columns:
    text_col = 'Review'
    print(f"\n{'─' * 80}")
    print(f"✅ RECOMENDACIÓN: Usar columna 'Review' para el modelo")
    print(f"   (contiene las reseñas completas con más contexto)")
    print(f"{'─' * 80}")
elif 'Title' in df.columns:
    text_col = 'Title'
    print(f"\n{'─' * 80}")
    print(f"⚠️  Solo disponible columna 'Title' (títulos cortos)")
    print(f"{'─' * 80}")

# ==============================================================================
# 5. RECOMENDACIONES INICIALES
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("5️⃣  RECOMENDACIONES INICIALES")
print(f"{'=' * 80}")

print(f"\n💡 ESTRATEGIA SUGERIDA PARA POLARIDAD:")
if imbalance_ratio > 20:
    print(f"  1. ❌ EVITAR SMOTE en embeddings (espacio muy dimensional)")
    print(f"  2. ✅ Focal Loss (ideal para desbalanceo extremo)")
    print(f"  3. ✅ Class weights agresivos")
    print(f"  4. ✅ Undersampling moderado de clases mayoritarias (4-5)")
    print(f"  5. ⚠️  Considerar agrupar clases: [1-2], [3], [4-5] si el negocio lo permite")
elif imbalance_ratio > 10:
    print(f"  1. ✅ Class weights")
    print(f"  2. ✅ Focal Loss o Weighted Cross-Entropy")
    print(f"  3. 🤔 Undersampling moderado de clase mayoritaria")
    print(f"  4. ❌ SMOTE NO recomendado en embeddings")
elif imbalance_ratio > 5:
    print(f"  1. ✅ Class weights (probablemente suficiente)")
    print(f"  2. 🤔 Focal Loss si class weights no funciona")
else:
    print(f"  1. ✅ Class weights ligeros")
    print(f"  2. ✅ O ninguna técnica (modelo puede aprender naturalmente)")

print(f"\n💡 ESTRATEGIA SUGERIDA PARA TIPO DE ATRACCIÓN:")
if type_imbalance > 5:
    print(f"  1. ✅ Class weights")
    print(f"  2. 🤔 Undersampling leve")
elif type_imbalance > 2:
    print(f"  1. ✅ Class weights")
else:
    print(f"  1. ✅ No requiere balanceo especial")

print(f"\n💡 ARQUITECTURA SUGERIDA:")
print(f"  • Opción 1 (RECOMENDADA): Multi-task learning - Un modelo con 2 cabezas")
print(f"    - Comparte embeddings entre tareas")
print(f"    - Más eficiente y puede mejorar generalización")
print(f"  • Opción 2: Dos modelos separados")
print(f"    - Más simple de implementar y debuggear")
print(f"    - Permite optimización independiente")

print(f"\n💡 MÉTRICAS DE EVALUACIÓN:")
print(f"  Para Polaridad: F1-Macro, Confusion Matrix, Recall por clase")
print(f"  Para Tipo: F1-Macro o Weighted (depende de importancia de clases)")

print(f"\n{'=' * 80}")
print("✅ ANÁLISIS COMPLETADO")
print(f"{'=' * 80}\n")

# Guardar información para siguiente script
info = {
    'text_col': text_col,
    'polarity_col': polarity_col,
    'type_col': type_col,
    'total_samples': len(df),
    'polarity_imbalance': imbalance_ratio,
    'type_imbalance': type_imbalance,
    'polarity_distribution': polarity_counts.to_dict(),
    'type_distribution': type_counts.to_dict()
}

import json
with open('analysis_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print("📁 Información guardada en: analysis_info.json")
