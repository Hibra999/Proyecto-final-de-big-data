import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. CONFIGURACIÓN DE RUTAS
# ---------------------------------------------------------
base_path = "/home/bigdata/Documentos/proyectoFinalBD/proyecto_final/"
archivo_resultados = base_path + "output/resultados_matriz_confusion.csv"
carpeta_imagenes = base_path + "images/"

# Crear carpeta para las imágenes si no existe
if not os.path.exists(carpeta_imagenes):
    os.makedirs(carpeta_imagenes)

print(f">>> Cargando resultados desde: {archivo_resultados}")

try:
    # 2. CARGAR DATOS
    df = pd.read_csv(archivo_resultados)

    # 3. GENERAR MATRIZ DE CONFUSIÓN (Heatmap)
    # ---------------------------------------------------------
    # Pivotamos los datos para tener Formato Matriz
    matriz_pivot = df.pivot_table(index='label', 
                                  columns='prediction', 
                                  values='count', 
                                  fill_value=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(matriz_pivot, annot=True, fmt='g', cmap='Blues')
    
    plt.title('Matriz de Confusión: Realidad vs Predicción', fontsize=15)
    plt.xlabel('Predicción del Modelo (Índice)')
    plt.ylabel('Etiqueta Real')
    
    ruta_grafico1 = carpeta_imagenes + "grafico_1_matriz.png"
    plt.savefig(ruta_grafico1)
    print(f">>> Gráfico 1 guardado en: {ruta_grafico1}")
    plt.close()

    # 4. GRÁFICO DE DISTRIBUCIÓN DE ATAQUES
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    ataques_count = df.groupby('label')['count'].sum().sort_values(ascending=False)
    ataques_count.plot(kind='bar', color='teal')
    
    plt.title('Distribución de Tipos de Tráfico Detectados', fontsize=15)
    plt.ylabel('Cantidad de Registros')
    plt.xticks(rotation=45)
    
    ruta_grafico2 = carpeta_imagenes + "grafico_2_distribucion.png"
    plt.tight_layout()
    plt.savefig(ruta_grafico2)
    print(f">>> Gráfico 2 guardado en: {ruta_grafico2}")
    plt.close()

    print("\n>>> ¡VISUALIZACIÓN COMPLETADA! Revisa la carpeta 'images'.")

except Exception as e:
    print(f"!!! ERROR: {e}")
