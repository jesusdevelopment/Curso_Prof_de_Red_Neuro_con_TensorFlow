# %% [markdown]
## Importación de Bibliotecas

from keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
import os
# %% [markdown]
## Carga del Dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
# %% [markdown]
## Exploración de los datos
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_train[1,1])
# %% [markdown]
## Graficación
plt.imshow(x_train[25])
print(y_train[25])  
# %% [markdown]
## Obtención de Etiquetas
base_path = "/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow"
data_dir = os.path.join(base_path, "data", "datacifar100")
local = os.path.join(data_dir, "cifar100_labels.json")
# Esto te dirá exactamente dónde está parado Python en este segundo
print(f"Ruta de ejecución actual: {os.getcwd()}")
print(f"Intentando crear: {data_dir}")
# %% [markdown]
## 2. CREAR LA CARPETA (Paso crítico)
# Usamos exist_ok=True para que no de error si ya existe
os.makedirs(data_dir, exist_ok=True)
# 3. Verificamos que la carpeta realmente se creó en el sistema
if os.path.isdir(data_dir):
    print(f"Directorio confirmado en: {data_dir}")

    ## 4. Descarga: Usamos comillas dobles alrededor de la ruta por si hay espacios
    url_etiquetas = "https://storage.googleapis.com/platzi-tf2/cifar100_labels.json"
    !wget --no-check-certificate "{url_etiquetas}" -O "{local}"
    
    ## 5. Extracción
    if os.path.exists(local):
        print(f"¡Éxito! Archivo descargado en: {local}")
    else:
        print("ERROR: wget no pudo guardar el archivo. Verifica que la ruta base_path sea correcta.")
else:
    print(f"ERROR: No se pudo crear el directorio {data_dir}. Revisa permisos.")

# %% [markdown]
## 6. Cargar las etiquetas
# %%
import json

with open(local, "r") as f:
    fine_labels = json.load(f)

# Recordatorio: y_train es (50000, 1), necesitamos acceder al índice [0]
indice_ejemplo  = 25
id_clase = y_train[indice_ejemplo][0]
nombre_clase = fine_labels[id_clase]
print(f"✅ Éxito: La imagen {indice_ejemplo} pertenece a la clase: {nombre_clase}")
# %% [markdown]
## 7. Graficación
# Visualizamos la imagen para confirmar que todo coincide.

plt.figure(figsize=(2,2))
plt.imshow(x_train[indice_ejemplo])
plt.title(f"Clase: {nombre_clase}")
plt.axis('off')
plt.show()
# %% 