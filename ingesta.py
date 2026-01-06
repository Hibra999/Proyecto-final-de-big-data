import findspark
findspark.init()

import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. CONFIGURACIÓN DE RUTAS (Ruta absoluta de tu Dell G15)
# ---------------------------------------------------------
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
base_path = "file:///home/bigdata/Documentos/proyectoFinalBD/proyecto_final/"
ruta_entrada = base_path + "output/iot23_procesado.parquet"
ruta_salida_csv = "/home/bigdata/Documentos/proyectoFinalBD/proyecto_final/output/resultados_matriz_confusion.csv"

try:
    # 2. INICIAR SESIÓN (Aumentamos memoria para los 25M de registros)
    # ---------------------------------------------------------
    spark = SparkSession.builder \
        .appName("IoT-23_ML_Gael") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()

    print(">>> Sesión iniciada. Cargando 25 millones de registros...")
    df = spark.read.parquet(ruta_entrada)

    # 3. PREPARACIÓN DE CARACTERÍSTICAS (Feature Engineering)
    # ---------------------------------------------------------
    # Columnas que el modelo usará para aprender
    cols_categoricas = ["proto", "service", "conn_state"]
    cols_numericas = ["id_resp_p", "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

    stages = []

    # Indexar categorías (Texto a números)
    for col_cat in cols_categoricas:
        indexer = StringIndexer(inputCol=col_cat, outputCol=col_cat + "_index", handleInvalid="keep")
        stages.append(indexer)

    # Indexar la etiqueta (Malware vs Benigno)
    label_indexer = StringIndexer(inputCol="label", outputCol="label_index", handleInvalid="skip")
    stages.append(label_indexer)

    # VectorAssembler: Junta todo en un solo vector
    input_cols = [c + "_index" for c in cols_categoricas] + cols_numericas
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    stages.append(assembler)

    # 4. DEFINICIÓN DEL MODELO
    # ---------------------------------------------------------
    # Usamos Random Forest. numTrees=20 para que no tarde horas en tu laptop.
    rf = RandomForestClassifier(featuresCol="features", labelCol="label_index", numTrees=20, maxDepth=10)
    stages.append(rf)

    # 5. ENTRENAMIENTO
    # ---------------------------------------------------------
    # Dividimos: 70% entrenamiento, 30% prueba
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
    
    print(">>> Entrenando el modelo... Esto tomará unos minutos por el volumen de datos.")
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(train_data)

    # 6. PREDICCIÓN Y EVALUACIÓN
    # ---------------------------------------------------------
    predictions = model.transform(test_data)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f">>> EXACTITUD DEL MODELO (Accuracy): {accuracy:.2%}")

    # 7. EXPORTAR RESULTADOS PARA EL PASO 3 (Visualización)
    # ---------------------------------------------------------
    print(">>> Generando matriz de confusión...")
    confusion_matrix = predictions.groupBy("label", "prediction").count().toPandas()
    
    # Guardar a CSV (usamos Pandas para que sea un solo archivo fácil de leer)
    confusion_matrix.to_csv(ruta_salida_csv, index=False)
    print(f">>> Resultados guardados en: {ruta_salida_csv}")

    spark.stop()

except Exception as e:
    print(f"!!! ERROR: {e}")
