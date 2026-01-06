import findspark
findspark.init()

import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. CONFIGURACIÓN DE RUTAS
# ---------------------------------------------------------
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
base_path = "file:///home/bigdata/Documentos/proyectoFinalBD/proyecto_final/"
ruta_entrada = base_path + "output/iot23_procesado.parquet"
ruta_salida_csv = "/home/bigdata/Documentos/proyectoFinalBD/proyecto_final/output/resultados_matriz_confusion.csv"

try:
    # 2. INICIAR SESIÓN (Ajustamos memoria a 6g para dejarle aire al sistema operativo)
    # ---------------------------------------------------------
    spark = SparkSession.builder \
        .appName("IoT-23_ML_Gael") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()

    print(">>> Cargando datos...")
    df_completo = spark.read.parquet(ruta_entrada)

    # --- CAMBIO CRÍTICO: MUESTREO ---
    # Tomamos el 10% de los datos (0.1) de forma aleatoria. 
    # 2.5 millones de filas siguen siendo 'Big Data' pero manejables.
    print(">>> Realizando muestreo del 10% para evitar errores de memoria...")
    df = df_completo.sample(withReplacement=False, fraction=0.1, seed=42)
    # --------------------------------

    # 3. FEATURE ENGINEERING
    # ---------------------------------------------------------
    cols_categoricas = ["proto", "service", "conn_state"]
    cols_numericas = ["id_resp_p", "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

    stages = []
    for col_cat in cols_categoricas:
        indexer = StringIndexer(inputCol=col_cat, outputCol=col_cat + "_index", handleInvalid="keep")
        stages.append(indexer)

    label_indexer = StringIndexer(inputCol="label", outputCol="label_index", handleInvalid="skip")
    stages.append(label_indexer)

    input_cols = [c + "_index" for c in cols_categoricas] + cols_numericas
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    stages.append(assembler)

    # 4. MODELO (Reducimos ligeramente la complejidad)
    # ---------------------------------------------------------
    rf = RandomForestClassifier(featuresCol="features", labelCol="label_index", numTrees=15, maxDepth=8)
    stages.append(rf)

    # 5. ENTRENAMIENTO
    # ---------------------------------------------------------
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
    
    print(f">>> Entrenando con aproximadamente {train_data.count()} registros...")
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(train_data)

    # 6. PREDICCIÓN Y EVALUACIÓN
    # ----------------------------
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f">>> EXACTITUD FINAL (Accuracy): {accuracy:.2%}")

    # 7. EXPORTAR RESULTADOS
    # ---------------------------------------------------------------
    confusion_matrix = predictions.groupBy("label", "prediction").count().toPandas()
    confusion_matrix.to_csv(ruta_salida_csv, index=False)
    print(f">>> Resultados listos en: {ruta_salida_csv}")

    spark.stop()

except Exception as e:
    print(f"!!! ERROR: {e}")
