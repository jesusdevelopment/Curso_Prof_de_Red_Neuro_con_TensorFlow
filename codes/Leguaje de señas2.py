# %%
import os
import json
 #from regex import F
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import string

# %% [markdown]
## 1. Definimos la Ruta Base del Proyecto

base_path = "/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow"
data_dir = os.path.join(base_path, "data")
local_zip = os.path.join(data_dir, "sign-language-img.zip")
extract_dir = os.path.join(data_dir, "sign-language-img")
# %% [markdown]
## 2. CREAR LA CARPETA (Paso crítico)
# Usamos exist_ok=True para que no de error si ya existe
os.makedirs(data_dir, exist_ok=True)
# 3. Verificamos que la carpeta realmente se creó en el sistema
if os.path.isdir(data_dir):
    print(f"Directorio confirmado en: {data_dir}")

    ## 4. Descarga: Usamos comillas dobles alrededor de la ruta por si hay espacios
    !wget --no-check-certificate "https://storage.googleapis.com/platzi-tf2/sign-language-img.zip" -O "{local_zip}"
    
    ## 5. Extracción
    if os.path.exists(local_zip):
        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
            print(f"¡Éxito! Archivos extraídos en: {extract_dir}")
    else:
        print("ERROR: wget no pudo guardar el archivo. Verifica que la ruta base_path sea correcta.")
else:
    print(f"ERROR: No se pudo crear el directorio {data_dir}. Revisa permisos.")

# %% [markdown]
## Carga de Datos Training y Test

train_dir=os.path.join(extract_dir, 'Train')
test_dir=os.path.join(extract_dir, 'Test') # Assuming test and validation are the same for now, or will be split later


# %% [markdown]
## Generadores

train_ds=tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    image_size=(64,64),
    interpolation='nearest',
    batch_size=128,    
    shuffle=True,
    validation_split=0.2,
    subset='training',
    seed=42
)

test_ds=tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    image_size=(64,64),
    interpolation='nearest',
    batch_size=128,
    shuffle=False,
)


validation_ds=tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    image_size=(64,64),
    interpolation='nearest',
    batch_size=128,
    shuffle=True,
    validation_split=0.2,
    subset='validation',
    seed=42
)
# %% 
classes=[char for char in string.ascii_uppercase if char !="J" if char !="Z"]
print(classes)
print(len(classes))
# %% [markdown]
## Graficación de Muestra de 5 imágenes

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(5):
        ax=plt.subplot(1,5,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classes[np.argmax(labels[i])])
        plt.axis("off")
plt.show()
# %% [markdown]
## Creación del Modelo
