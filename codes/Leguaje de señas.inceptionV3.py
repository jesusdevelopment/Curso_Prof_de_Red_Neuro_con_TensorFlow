# %%
!pip install -q -U keras-tuner
import os
import json
 #from regex import F
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO  
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import string
import zipfile
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint


# %% [markdown]
## 1. Definimos la Ruta Base del Proyecto

base_path = "/home/jesusr/Proyectos_Deep_Learning/Curso_Prof_de_Red_Neuro_con_TensorFlow"
data_dir = os.path.join(base_path, "data")
local_zip = os.path.join(data_dir, "sign-language-img.zip")
extract_dir = os.path.join(data_dir, "sign-language-img")
# %% [markdown]
## 2. CREAR LA CARPETA Y DESCARGA DEL DATASET
# Usamos exist_ok=True para que no de error si ya existe
os.makedirs(data_dir, exist_ok=True)

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
##3. Rutas Training y Test

train_dir=os.path.join(extract_dir, 'Train')
test_dir=os.path.join(extract_dir, 'Test') # Assuming test and validation are the same for now, or will be split later


# %% [markdown]
##4. Cargadores de Datos (Datasets de TensorFlow)

train_ds=tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    image_size=(150,150),
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
    color_mode='rgb',
    image_size=(150,150),
    interpolation='nearest',
    batch_size=128,
    shuffle=False,
)


validation_ds=tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    image_size=(150,150),
    interpolation='nearest',
    batch_size=128,
    shuffle=True,
    validation_split=0.2,
    subset='validation',
    seed=42
)
# %% [markdown]
##5. Mapeo de Clases
classes=[char for char in string.ascii_uppercase if char !="J" if char !="Z"]
print(classes)
print(len(classes))
# %% [markdown]
##6. Graficación de Muestra de 5 imágenes

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(5):
        ax=plt.subplot(1,5,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classes[np.argmax(labels[i])])
        plt.axis("off")
plt.show()
# %% [markdown]
##7. Carga de Modelo InceptionV3

pretrained_model = tf.keras.applications.InceptionV3(input_shape=(150, 150, 3),
                                               include_top=False,
                                               weights='imagenet')
pretrained_model.trainable = False
pretrained_model.summary()

# Hacemos el corte arquitectónico en la capa mixed7
last_layer = pretrained_model.get_layer("mixed7")

# Encapsulamos la sub-red de Inception desde su entrada original hasta mixed7
inception_submodel = tf.keras.Model(
    inputs=pretrained_model.input, 
    outputs=last_layer.output
)

print("Shape of last_output:", inception_submodel.output.shape)

# %% [markdown]
## 8. Construcción del Modelo con la API Funcional (Incluye Normalización)

# Nodo de Entrada para imágenes crudas (píxeles de 0 a 255)
inputs = tf.keras.Input(shape=(150, 150, 3))

# Capa de Rescaling configurada para la matemática exacta de InceptionV3 (-1 a 1)
x = tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1.0)(inputs)
# Pasamos los datos normalizados a través de nuestro sub-modelo pre-entrenado
x = inception_submodel(x)
# Añadimos las capas densas personalizadas de salida
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation = "relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(classes), activation = "softmax")(x)

# Enlazamos la arquitectura funcional completa
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# %% [markdown]
##9. Compilación del Modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
# %% [markdown]
##10. Entrenamiento del Modelo

pretrained_history = model.fit(train_ds,
                    epochs=5,
                    validation_data=validation_ds)


# %% [markdown]
##11. Gráficas de Exactitud y Pérdida

acc = pretrained_history.history['accuracy']
val_acc = pretrained_history.history['val_accuracy']
loss = pretrained_history.history['loss']
val_loss = pretrained_history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# %% [markdown]
##12. Evaluación del Modelo

loss, accuracy = model.evaluate(test_ds)

print(f"Loss en el conjunto de prueba: {loss:.4f}")
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")


# %% [markdown]
##13. Guardar el Modelo Completo

model.save("inceptionv3_sign_language_model.h5")
print("Modelo guardado como inceptionv3_sign_language_model.h5")


# %% [markdown]
## Cargar el Modelo y Realizar Predicciones

# Cargar el modelo
loaded_model = tf.keras.models.load_model("inceptionv3_sign_language_model.h5")

# Función para preprocesar una imagen antes de la predicción
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Crear un batch
    return img_array

# Realizar una predicción con una imagen de ejemplo
# Asegúrate de tener una imagen de prueba en tu directorio o descarga una.
# Por ejemplo, puedes usar una imagen del conjunto de prueba.
# test_image_path = "/path/to/your/test_image.jpg"
# preprocessed_image = preprocess_image(test_image_path)
# predictions = loaded_model.predict(preprocessed_image)
# predicted_class = np.argmax(predictions[0])
