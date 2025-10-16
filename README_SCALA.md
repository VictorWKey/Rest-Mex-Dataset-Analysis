# Usar los Modelos en Scala

## Archivos Generados

Después de ejecutar el notebook `02_model_pipeline.ipynb`, se generan los siguientes archivos:

### 1. Datasets CSV con Embeddings
- **`embeddings_complete.csv`**: Dataset completo con embeddings (384 dimensiones) + labels
  - Columnas: `emb_0`, `emb_1`, ..., `emb_383`, `polarity`, `type`, `review_text`
  - Filas: ~190,000 registros balanceados

- **`embeddings_with_split.csv`**: Dataset con columna adicional `split` (train/test)
  - Mismas columnas + `split` ('train' o 'test')

### 2. Modelos Entrenados
- **`model_polarity.pkl`**: Modelo Gradient Boosting para polaridad (5 clases: 1-5)
- **`model_type.pkl`**: Modelo Random Forest para tipo de atracción (3 clases: Hotel, Attractive, Restaurant)
- **`models_metadata.json`**: Metadata de los modelos (dimensiones, clases, mapeos)

---

## Cómo Usar en Scala

### Opción 1: Leer CSVs y Entrenar Nuevos Modelos en Spark MLlib

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{RandomForestClassifier, GBTClassifier}
import org.apache.spark.ml.feature.VectorAssembler

val spark = SparkSession.builder()
  .appName("RestMex Classifier")
  .master("local[*]")
  .getOrCreate()

// Leer CSV con embeddings
val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("embeddings_with_split.csv")

// Filtrar train/test
val trainDF = df.filter("split = 'train'")
val testDF = df.filter("split = 'test'")

// Crear vector de features (embeddings)
val embeddingCols = (0 until 384).map(i => s"emb_$i")
val assembler = new VectorAssembler()
  .setInputCols(embeddingCols.toArray)
  .setOutputCol("features")

val trainData = assembler.transform(trainDF)
val testData = assembler.transform(testDF)

// Entrenar modelo de polaridad con GBT
val gbt = new GBTClassifier()
  .setLabelCol("polarity")
  .setFeaturesCol("features")
  .setMaxIter(100)
  .setMaxDepth(7)

val polarityModel = gbt.fit(trainData)

// Predicciones
val predictions = polarityModel.transform(testData)
predictions.select("polarity", "prediction").show()

// Entrenar modelo de tipo con Random Forest
val rf = new RandomForestClassifier()
  .setLabelCol("type")
  .setFeaturesCol("features")
  .setNumTrees(200)
  .setMaxDepth(20)

// Convertir tipo a numérico primero
import org.apache.spark.ml.feature.StringIndexer
val indexer = new StringIndexer()
  .setInputCol("type")
  .setOutputCol("type_indexed")
  .fit(trainData)

val trainIndexed = indexer.transform(trainData)
val testIndexed = indexer.transform(testData)

val typeModel = rf.setLabelCol("type_indexed").fit(trainIndexed)
val typePredictions = typeModel.transform(testIndexed)
```

---

### Opción 2: Cargar Modelos Python con MLeap (Avanzado)

Para usar los modelos `.pkl` directamente en Scala, necesitas convertirlos a formato MLeap:

#### Paso 1: Convertir modelos Python a MLeap (en Python)

```python
# Instalar: pip install mleap
from mleap.sklearn.ensemble import GradientBoostingClassifier
from mleap.sklearn.pipeline import Pipeline
import pickle

# Cargar modelo
with open('model_polarity.pkl', 'rb') as f:
    model = pickle.load(f)

# Exportar a MLeap
from mleap.pyspark.spark_support import SimpleSparkSerializer
# ... (proceso de conversión)
```

#### Paso 2: Cargar en Scala

```scala
import ml.combust.mleap.runtime.MleapContext.defaultContext
import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import resource._

val bundle = (for(bundleFile <- managed(BundleFile("jar:file:/path/to/model.zip"))) yield {
  bundleFile.loadMleapBundle().get
}).opt.get

val model = bundle.root
```

---

### Opción 3: Usar Breeze para Predicciones (Solo Inference)

Si solo necesitas hacer inferencia (no entrenar):

```scala
import breeze.linalg._

// Cargar embeddings
val data = csvread(new File("embeddings_complete.csv"), ',', skipLines = 1)

// Implementar lógica de Random Forest/GBT manualmente
// (más complejo, no recomendado)
```

---

## Recomendación: Opción 1 (Spark MLlib)

**La mejor opción es leer los CSVs y entrenar nuevos modelos en Spark MLlib:**

✅ **Ventajas:**
- Nativo en Scala/Spark
- Optimizado para Big Data
- Fácil integración con pipelines de Spark
- No requiere conversiones complejas

✅ **Los embeddings ya están calculados**, así que:
- No necesitas calcular embeddings en Scala (la parte más lenta)
- Solo entrenas el clasificador (rápido)
- Puedes usar Spark ML pipelines completos

---

## Ejemplo Completo en Scala

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object RestMexClassifier {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("RestMex Polarity Classifier")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Leer datos
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("embeddings_with_split.csv")

    // Split train/test
    val Array(trainDF, testDF) = df.randomSplit(Array(0.8, 0.2), seed = 42)

    // Crear features vector
    val embeddingCols = (0 until 384).map(i => s"emb_$i")
    val assembler = new VectorAssembler()
      .setInputCols(embeddingCols.toArray)
      .setOutputCol("features")

    val trainData = assembler.transform(trainDF)
    val testData = assembler.transform(testDF)

    // Entrenar modelo
    val gbt = new GBTClassifier()
      .setLabelCol("polarity")
      .setFeaturesCol("features")
      .setMaxIter(100)
      .setMaxDepth(7)
      .setSeed(42)

    println("Entrenando modelo de polaridad...")
    val model = gbt.fit(trainData)

    // Predicciones
    val predictions = model.transform(testData)

    // Evaluación
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("polarity")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val f1Score = evaluator.evaluate(predictions)
    println(s"F1-Score: $f1Score")

    // Guardar modelo
    model.save("models/polarity_model_scala")

    spark.stop()
  }
}
```

---

## Estructura de Archivos Final

```
Rest-Mex-Dataset-Analysis/
├── 02_model_pipeline.ipynb          # Notebook principal
├── embeddings_complete.csv          # 190K filas x 387 columnas
├── embeddings_with_split.csv        # Con columna 'split'
├── model_polarity.pkl               # Modelo Python (Gradient Boosting)
├── model_type.pkl                   # Modelo Python (Random Forest)
├── models_metadata.json             # Metadata
└── scala_classifier/                # Tu código Scala aquí
    ├── build.sbt
    └── src/main/scala/RestMexClassifier.scala
```

---

## Dependencias Scala (build.sbt)

```scala
name := "RestMex-Classifier"
version := "1.0"
scalaVersion := "2.12.15"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.0",
  "org.apache.spark" %% "spark-sql" % "3.3.0",
  "org.apache.spark" %% "spark-mllib" % "3.3.0"
)
```

---

¡Listo! Con estos archivos puedes trabajar completamente en Scala sin necesidad de Python. 🚀
