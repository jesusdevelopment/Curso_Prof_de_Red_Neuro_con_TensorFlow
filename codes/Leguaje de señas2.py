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
    image_size=(28,28),
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
    image_size=(28,28),
    interpolation='nearest',
    batch_size=128,
    shuffle=False,
)


validation_ds=tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    image_size=(28,28),
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
# %% [markdown]
## Optimización del Rendimiento de los Datasets

# Mantenemos esto porque es una buena práctica para acelerar la lectura de datos
#AUTOTUNE = tf.data.AUTOTUNE
#train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
#test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %% [markdown]
## Creación del Modelo Secuencial Base (Perceptrón Multicapa - Solo Redes Densas)

model = tf.keras.models.Sequential([
    # 1. Preprocesamiento: Normalizamos los píxeles de [0, 255] a [0, 1]
    tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)),
    
    # 2. APLANADO (FLATTEN) - ¡El paso más importante aquí!
    # Toma la matriz de 28x28 y la estira en un solo vector de 784 números.
    # Las capas Densas solo entienden información en 1 dimensión.
    tf.keras.layers.Flatten(),
    
    # 3. Capas Ocultas (Red Neuronal Profunda Tradicional)
    # Empezamos con muchas neuronas y vamos reduciendo como un embudo
    #tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    
    # Apagamos el 30% de las neuronas al azar en cada época para evitar que 
    # el modelo memorice (overfitting)
    #tf.keras.layers.Dropout(0.3), 
    
    tf.keras.layers.Dense(128, activation='relu'),
    
    # 4. Capa de Salida
    # 24 neuronas (una para cada letra de tu dataset).
    # Softmax convierte los resultados en porcentajes de probabilidad que suman 1 (100%)
    tf.keras.layers.Dense(24, activation='softmax')
])

# Veamos cuántos parámetros (pesos) va a tener que aprender este modelo
model.summary()

# %% [markdown]
## Compilación del Modelo

model.compile(
    optimizer='adam',
    # categorical_crossentropy porque usaste label_mode='categorical' en tus generadores
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# %% [markdown]
## Entrenamiento del Modelo

epochs = 20

print("Iniciando el entrenamiento de la Red Densa...")
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs
)

# %% [markdown]
## Evaluación Final en el Set de Prueba

loss, accuracy = model.evaluate(test_ds)
print(f"\nPrecisión final en datos no vistos (Test): {accuracy * 100:.2f}%")
# %%[markdown]
## Reporte de Clasificación

def visualizacion_resultados(history, model_name):
    # Gráfica de precisión y pérdida
    plt.figure(figsize=(12, 5))

    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title(f'Precisión del Modelo {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title(f'Pérdida del Modelo {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()
    
visualizacion_resultados(history, "Red Densa")

# %% [markdown]
## Callbacks
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
            if logs.get('val_accuracy') > 0.95:
                print("\n¡Precisión de validación alcanzada > 95%! Cancelando entrenamiento.")
                self.model.stop_training = True
# %% [markdown]
## Instanciamiento y Entrenamiento
#1. Volvemos a definir el modelo para que sus pesos empiecen de cero
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(24, activation='softmax')
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callback=CustomCallback()

history_callback=model2.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[callback]
)
visualizacion_resultados(history_callback, "Red Densa con Callback")
# %% [markdown]
## Keras Tunning

def constructor_modelos(hp):
    model = tf.keras.models.Sequential()
    
    # Capas fijas (Extracción de características)
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(75, (3,3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Flatten())

    # --- ESPACIO DE BÚSQUEDA ---
    # 1. Ajuste dinámico de neuronas
    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation="relu", 
                           kernel_regularizer=regularizers.l2(1e-5)))
    
    # 2. Ajuste dinámico de Dropout
    hp_dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(rate=hp_dropout))
    
    # 23. Capas estables
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(len(classes), activation="softmax"))

    # 3. Elección de Learning Rate
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate), loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model

# %% [markdown]
## Tunning

tuner = kt.Hyperband(
    constructor_modelos,
    objective = "val_accuracy",
    max_epochs = 20,
    factor = 3,
    directory = "models/",
    project_name = "platzi-tunner"
)

tuner.search(train_ds, epochs=20, validation_data= validation_ds)
best_hps= tuner.get_best_hyperparameters(num_trials=1)[0]

# %% [markdown]
## Análisis de Resultados
tuner.results_summary()

print(best_hps.get("units"))
print(best_hps.get("learning_rate"))

# %% [markdown]
## Mejor modelo

hypermodel = tuner.hypermodel.build(best_hps)
callback_early = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 3, mode = "auto")
history_hypermodel = hypermodel.fit(
    train_ds,
    epochs = 20,
    callbacks = [callback_early],
    validation_data = validation_ds
)


  # %% [markdown]
## Sumary
hypermodel.summary()
# %%
loss, accuracy = hypermodel.evaluate(test_ds)
# %%
print(f"Loss: {loss}, Accuracy: {accuracy}")

# %% [markdown]
## Guardando Arquitectura

model_json = hypermodel.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# %%
print(model_json)

# %% [markdown]
## Guardando Pesos
hypermodel.save_weights("model.h5")

# %% [markdown]
## Carga del Modelo

# %%
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# %%
loaded_model.load_weights("model.h5")

# %%
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
loss, accuracy = loaded_model.evaluate(test_ds)
print(f"Loss del modelo cargado: {loss}, Accuracy del modelo cargado: {accuracy}")
# %%
