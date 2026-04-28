# %% [markdown]
## Importación de Bibliotecas
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
## Train y Test
train="/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/brain_tumors/dataset_extraido/Training"
test="/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/brain_tumors/dataset_extraido/Testing"


# Definimos los parámetros comunes
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Cargamos el set de ENTRENAMIENTO (80%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    extract_dir + '/Training',
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Cargamos el set de VALIDACIÓN (20%)
val_ds = tf.keras.utils.image_dataset_from_directory(
    extract_dir + '/Training',
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Cargamos el set de TEST
test_ds = tf.keras.utils.image_dataset_from_directory(
    extract_dir + '/Testing',
    subset="testing",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)