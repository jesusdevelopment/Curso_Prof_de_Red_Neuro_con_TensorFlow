# %% [markdown]
## Importación de Bibliotecas
from turtle import color

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import zipfile
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from io import BytesIO


# %% [markdown]
## Carga del Dataset

# Esto te dirá exactamente dónde está parado Python en este seg
# %% [markdown]
## Carga del Dataset

base_path = "/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow"
data_dir = os.path.join(base_path, "data", "brain_tumors")
local = os.path.join(data_dir, "databasesLoadData.zip")
extract_dir = os.path.join(data_dir, "dataset_extraido")

# CRÍTICO: Primero creamos la carpeta, de lo contrario wget no tendrá dónde guardar
os.makedirs(data_dir, exist_ok=True)

if os.path.isdir(data_dir):
    print(f"Directorio confirmado en: {data_dir}")

    # Corregido: La URL debe ir entre comillas
    url = "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset"
    
    # Descarga
    print("Iniciando descarga...")
    !wget --no-check-certificate "{url}" -O "{local}"
        
    ## 5. Extracción 
    if os.path.exists(local):
        print(f"¡Éxito! Archivo descargado en: {local}")
        
        # Lógica de extracción que faltaba
        with zipfile.ZipFile(local, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Dataset extraído en: {extract_dir}")
    else:
        print("ERROR: wget no pudo guardar el archivo. Verifica tu conexión o el enlace.")
else:
    print(f"ERROR: No se pudo crear el directorio {data_dir}.")
        
# %% [markdown]
## EDA del dataset
# Balanceo de Clases

train_dir = os.path.join(extract_dir, 'Training')
test_dir = os.path.join(extract_dir, 'Testing')

sets = ['Training', 'Testing']
MIS_CLASES = ['glioma', 'meningioma', 'notumor', 'pituitary']
stats = []

for dataset_type in sets:
    for label in MIS_CLASES:
        # Construimos la ruta dinámicamente para Training y Testing
        path = os.path.join(extract_dir, dataset_type, label)
        if os.path.exists(path):
            count = len(os.listdir(path))
            stats.append({
                'label': label, 
                'count': count, 
                'dataset': dataset_type  # Nueva columna para diferenciar
            })

# Crear el DataFrame
df_stats = pd.DataFrame(stats)

# Graficar usando 'hue' para separar por Training/Testing
plt.figure(figsize=(12, 6))
sns.barplot(x='label', y='count', hue='dataset', data=df_stats)

plt.title('Comparación de Distribución de Clases: Entrenamiento vs. Test')
plt.xlabel('Tipo de Tumor')
plt.ylabel('Cantidad de Imágenes')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# %% [markdown]
## Train y Test
train="/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/brain_tumors/dataset_extraido/Training"
test="/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/brain_tumors/dataset_extraido/Testing"


# Definimos los parámetros comunes
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Cargamos el set de ENTRENAMIENTO (80%)
# Cargamos el set de ENTRENAMIENTO (80% de la carpeta Training)
train_ds = tf.keras.utils.image_dataset_from_directory(
    extract_dir + '/Training',
    validation_split=0.2,
    class_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
    color_mode='rgb',
    subset="training",
    seed=42,
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE
)

# Cargamos el set de VALIDACIÓN (20% de la carpeta Training)
val_ds = tf.keras.utils.image_dataset_from_directory(
    extract_dir + '/Training',
    validation_split=0.2,
    class_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
    color_mode='rgb',
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Cargamos el set de TEST (Carpeta Testing completa)
test_ds = tf.keras.utils.image_dataset_from_directory(
    extract_dir + '/Testing',
    class_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
    color_mode='rgb',
    shuffle=False,      # ¡Importante! Mantiene el orden para la evaluación final
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
    # Nota: No usamos 'subset' ni 'validation_split' aquí
)
# %% [markdown]
# Visualización de 5 imágenes

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1): # Tomamos un batch del dataset de entrenamiento
    for i in range(5): # Iteramos sobre las primeras 5 imágenes del batch
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8")) # Mostramos la imagen
        plt.title(train_ds.class_names[labels[i]]) # Establecemos el título con el nombre de la clase
        plt.axis("off") # Ocultamos los ejes
plt.show()


# %% [markdown]
# Preprocesamiento de Datos con capas dentro del modelo
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2)
])

# Normalización de las imágenes
preprocess_input = tf.keras.layers.Rescaling(1./255)

# %% [markdown]
# Configuración del rendimiento
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %% [markdown]
# Creación del modelo base (MobileNetV2)

# Cargar el modelo pre-entrenado MobileNetV2 sin las capas superiores
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                               include_top=False,
                                               weights='imagenet')
