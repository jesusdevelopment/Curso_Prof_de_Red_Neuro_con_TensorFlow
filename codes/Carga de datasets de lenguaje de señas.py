
# %%
import os
import zipfile
import json
from regex import F
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import base64
# %% [markdown]
## 1. Definimos la Ruta Base del Proyecto

base_path = "/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow"
data_dir = os.path.join(base_path, "data")
local_zip = os.path.join(data_dir, "databasesLoadData.zip")
extract_dir = os.path.join(data_dir, "dataset_extraido")
# %% [markdown]
## 2. CREAR LA CARPETA (Paso crítico)
# Usamos exist_ok=True para que no de error si ya existe
os.makedirs(data_dir, exist_ok=True)
# 3. Verificamos que la carpeta realmente se creó en el sistema
if os.path.isdir(data_dir):
    print(f"Directorio confirmado en: {data_dir}")

    ## 4. Descarga: Usamos comillas dobles alrededor de la ruta por si hay espacios
    !wget --no-check-certificate "https://storage.googleapis.com/platzi-tf2/databasesLoadData.zip" -O "{local_zip}"
    
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
## 6. Ruta al archivo JSON
url_jason='/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/dataset_extraido/sign_mnist_json/data.json' 
# %% [markdown]
## 7. Cargar el archivo JSON
data_json = []

## 6. Abrimos el archivo de forma estándar
with open(url_jason, 'r', encoding='utf-8') as f:
    # Iteramos directamente sobre el archivo, línea por línea
    for line in f:
        # Usamos json.loads() (con 's' al final, para Strings) en cada línea
        data_json.append(json.loads(line.strip()))

print(f"✅ ¡Éxito! Se cargaron {len(data_json)} registros/imágenes encontradas.")

## 7. Imprimir el primer registro para ver su estructura
print("\nMuestra del primer registro:")
print(data_json[0])
## 8. Procesar cada registro para obtener la imagen y su etiqueta
images_jason = []

for data in data_json:
  response = requests.get(data['content'])
  img = np.asarray(Image.open(BytesIO(response.content)))
  images_jason.append([img, data["label"]])

plt.imshow(images_jason[0][0].reshape(28,28))
print(images_jason[0][1])
# %% [markdown]
## 9. Ruta al archivo JSON con la imagen codificada en base64
url_base64='/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/dataset_extraido/sign_mnist_base64/data.json'
# %% [markdown]
with open(url_base64) as f:
  data = json.load(f)
## Imprimir Registro para verificar que se cargó correctamente
data
# %% [markdown]
## 10. Decodificar la imagen en base64 y guardarla como archivo PNG
base64_img_bytes = data['b'].encode('utf-8')
path_img = "/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/decoded_images.png"
with open(path_img, "wb") as file_to_save:
  decoded_image_data = base64.decodebytes(base64_img_bytes)
  file_to_save.write(decoded_image_data)
# %% [markdown]
## 11. Cargar la imagen decodificada usando PIL y mostrarla
img=Image.open(path_img)
img
# %% [markdown]
## 12. Cargar una imagen directamente desde la ruta del dataset usando PIL
img=Image.open('/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/dataset_extraido/pixeles.png')
img
# %% [markdown]
## 13. Cargar los archivos CSV de entrenamiento y prueba usando pandas
train=pd.read_csv('/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/dataset_extraido/sign_mnist_train/sign_mnist_train.csv')
test=pd.read_csv('/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/dataset_extraido/sign_mnist_test/sign_mnist_test.csv')
# %%
train.head()
# %%
train.shape
# %%
train_labels=train['label'].values
# %%
train_labels
# %%
train_labels.shape
# %%
train.drop('label', axis=1, inplace=True)
# %%
train.head()
# %%
train_images=train.values
# %%
train_images.shape
# %%
train_images
# %%
plt.imshow(train_images[0].reshape(28,28))
# %%
re=train_images.reshape(27455,28,28)
# %%
re
# %% [markdown]
## 14. Carga del Dataset CSV para Limpiar
train_to_clean=pd.read_csv('/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow/data/dataset_extraido/sign_mnist_train/sign_mnist_train_clean.csv')
print(train_to_clean.shape)
train_to_clean.head()
train_to_clean.info()
# %% [markdown]
## Creación de y_train y_test
y_train_to_clean=train_to_clean['label']
train_to_clean.drop('label', axis=1, inplace=True)
y_test=test['label'].values
test.drop('label', axis=1, inplace=True)
# %% [markdown]
## Análisis de datos
train_to_clean.dtypes
print(train_to_clean.info())
train_to_clean.nunique()
# %%
y_train_to_clean.dtypes
print(y_train_to_clean.info())
train_to_clean.head()
# %% [markdown]
## Valores nulos
y_train_to_clean.isna().sum()
# %%
train_to_clean.isna().value_counts()
# %%
train_to_clean.isna().sum()
# %%
train_to_clean.isna().sum().sum()
# %%
train_to_clean.isna().values.any()
# %%
train_to_clean.isna().any()
# %%
train_to_clean.dtypes
# %%  [markdown]
## Datos Duplicados
train_to_clean.duplicated().sum()
# %%
train_to_clean[train_to_clean.duplicated(keep=False)]
# %%
train_to_clean.drop_duplicates(keep=False, inplace=True)
# %% [markdown]
## Cambio de tipos de datos
train_to_clean=train_to_clean.astype('float32')
# %% [markdown]
## Normalización de los datos
train_to_clean=train_to_clean/255.0
test=test/255.0
# %%
train_to_clean.head()
test.head()
# %%

# %% [markdown]
