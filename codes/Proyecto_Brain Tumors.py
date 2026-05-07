# %% [markdown]
## Importación de Bibliotecas
!pip install imagehash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import zipfile
from PIL import Image
import glob   
from io import BytesIO
import hashlib

# %% [markdown]
## 1. Conexión con Google Drive y Carga de Datos Limpios

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. Configuración de rutas
ruta_rar = "/content/drive/MyDrive/Data/Curso Prof TensorFlow/dataset_extraido.rar"
extract_dir = "/content/dataset_trabajo"

# 3. Proceso de extracción con limpieza previa
if os.path.exists(ruta_rar):
    # Borramos si existía algo corrupto para empezar de cero
    if os.path.exists(extract_dir):
        !rm -rf "{extract_dir}"
    
    os.makedirs(extract_dir, exist_ok=True)
    
    print("📦 Extrayendo dataset... Por favor, espera a que aparezca el mensaje de éxito.")
    # -o+ (sobreescribir), -idq (modo silencioso para no llenar la consola)
    !unrar x -o+ -idq "{ruta_rar}" "{extract_dir}/"
    
    # --- VERIFICACIÓN DINÁMICA ---
    # Buscamos la carpeta 'Training' en cualquier nivel de profundidad
    print("🔍 Buscando carpetas de datos...")
    hallazgos = glob.glob(os.path.join(extract_dir, "**/Training"), recursive=True)
    
    if hallazgos:
        # Definimos las rutas reales basadas en el hallazgo
        base_real = os.path.dirname(hallazgos[0])
        train_dir = os.path.join(base_real, 'Training')
        test_dir = os.path.join(base_real, 'Testing')
        
        print(f"✅ ¡Éxito! Carpetas encontradas en: {base_real}")
        print(f"📍 Train: {train_dir}")
        print(f"📍 Test:  {test_dir}")
        
        # Sobreescribimos la variable para que el resto de tu código funcione
        extract_dir = base_real 
    else:
        print("❌ ERROR: El .rar se extrajo pero no contiene una carpeta llamada 'Training'.")
        print("Contenido extraído:")
        !ls -R "{extract_dir}" | head -n 10
else:
    print(f"❌ ERROR: No se encontró el archivo en Drive: {ruta_rar}")

# 4. Definir rutas para el resto del código
train_dir = os.path.join(extract_dir, 'Training')
test_dir = os.path.join(extract_dir, 'Testing')

# %% [markdown]
## EDA: Balanceo de Clases (Corregido para Tumores Cerebrales)

# 1. Definimos las carpetas y clases EXACTAS del dataset de tumores
sets = ['Training', 'Testing']
MIS_CLASES_BRAIN = ['glioma', 'meningioma', 'notumor', 'pituitary']
stats = []

for dataset_type in sets:
    # Construimos la ruta: /home/jesusr/.../dataset_extraido/Training (o Testing)
    set_path = os.path.join(extract_dir, dataset_type)
    
    if not os.path.exists(set_path):
        print(f"⚠️ Alerta: No se encontró la carpeta: {set_path}")
        continue

    for label in MIS_CLASES_BRAIN:
        path = os.path.join(set_path, label)
        if os.path.exists(path):
            count = len(os.listdir(path))
            stats.append({
                'label': label, 
                'count': count, 
                'dataset': dataset_type
            })
        else:
            print(f"❌ Carpeta de clase no encontrada: {path}")

# 2. Crear el DataFrame y Graficar
df_stats = pd.DataFrame(stats)

if df_stats.empty:
    print("FATAL: El DataFrame está vacío. Revisa las rutas de 'extract_dir'.")
else:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='label', y='count', hue='dataset', data=df_stats)
    plt.title('Distribución de Clases: Glioma, Meningioma, No Tumor y Pituitaria')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Detección de archivos Corruptos

from PIL import Image

def check_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify() # Verifica que el archivo no esté corrupto
                except (IOError, SyntaxError):
                    print(f'Archivo corrupto eliminado: {path}')
                    os.remove(path)

check_images(extract_dir)

# Detección de Archivos Duplicados (Hashing)

import imagehash
from PIL import Image

def eliminar_duplicados_visuales(directorio):
    hashes_vistos = {}
    duplicados_eliminados = 0
    
    for root, dirs, files in os.walk(directorio):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        # Genera un hash basado en la estructura visual, no en los bits
                        # dhash es rápido y muy efectivo para detectar copias
                        v_hash = imagehash.dhash(img)
                    
                    if v_hash in hashes_vistos:
                        print(f"Eliminando duplicado visual: {path}")
                        os.remove(path)
                        duplicados_eliminados += 1
                    else:
                        hashes_vistos[v_hash] = path
                except Exception as e:
                    print(f"No se pudo procesar {file}: {e}")
                
    print(f"¡Limpieza visual terminada! Se eliminaron {duplicados_eliminados} archivos.")

# Ejecutar en ambos
eliminar_duplicados_visuales(train_dir)
eliminar_duplicados_visuales(test_dir)

# Análisis Dimensional (Tamaños y Proporciones)


## Definimos los directorios a analizar
sets_to_analyze = ['Training', 'Testing']
colors = {'Training': 'blue', 'Testing': 'orange'}

plt.figure(figsize=(10, 6))

for dataset_type in sets_to_analyze:
    widths, heights = [], []
    current_path = os.path.join(extract_dir, dataset_type)
    
    # Recorremos todas las subcarpetas de cada set
    for root, dirs, files in os.walk(current_path):
        for file in files:
            if file.endswith('.jpg'):
                try:
                    with Image.open(os.path.join(root, file)) as im:
                        w, h = im.size
                        widths.append(w)
                        heights.append(h)
                except Exception as e:
                    print(f"Error al abrir {file}: {e}")
    
    # Graficamos cada set con un color y etiqueta diferente
    plt.scatter(widths, heights, alpha=0.3, label=dataset_type, color=colors[dataset_type])

plt.xlabel('Ancho (píxeles)')
plt.ylabel('Alto (píxeles)')
plt.title('Comparación de Dimensiones: Entrenamiento vs. Test')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Resumen numérico
# Corrección para contar todos los archivos dentro de las subcarpetas
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_test = sum([len(files) for r, d, files in os.walk(test_dir)])

print("--- RESUMEN DEL DATASET ---")
print(f"Límite máximo de dimensiones: {max(widths)} (ancho) x {max(heights)} (alto)")
print(f"Límite mínimo de dimensiones: {min(widths)} (ancho) x {min(heights)} (alto)")
print(f"Total de imágenes para entrenamiento: {total_train}")
print(f"Total de imágenes para prueba (Test): {total_test}")
print(f"Proporción de entrenamiento: {total_train / (total_train + total_test):.2f}")
print(f"Proporción de prueba: {total_test / (total_train + total_test):.2f}")

# Análisis de Intensidad de Píxeles

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def analizar_intensidades(extract_dir, sets=['Training', 'Testing'], sample_size=300):
    plt.figure(figsize=(12, 6))
    colores = {'Training': 'blue', 'Testing': 'orange'}
    
    print("Iniciando análisis de intensidades de píxel...")

    for dataset_type in sets:
        dataset_path = os.path.join(extract_dir, dataset_type)
        all_files = []
        
        # 1. Recolectar todas las rutas de imágenes
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jpg'):
                    all_files.append(os.path.join(root, file))
        
        # 2. Tomar una muestra aleatoria para no colapsar la RAM
        # 300 imágenes es estadísticamente suficiente para ver la distribución
        sampled_files = random.sample(all_files, min(sample_size, len(all_files)))
        
        # 3. Crear un acumulador de ceros para los 256 posibles valores de gris
        hist_acumulado = np.zeros((256, 1))
        
        # 4. Sumar el histograma de cada imagen al acumulador
        for img_path in sampled_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # calcHist es extremadamente rápido
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist_acumulado += hist
                
        # 5. Normalizar: Dividir entre el total de píxeles para tener porcentajes (0 a 1)
        # Esto permite comparar justamente aunque un set tenga más imágenes que otro
        hist_acumulado /= hist_acumulado.sum()
        
        # 6. Graficar como una línea continua
        plt.plot(hist_acumulado, color=colores[dataset_type], label=f'{dataset_type} (n={len(sampled_files)})', alpha=0.8)

    plt.title('Distribución de Intensidades de Píxel: Training vs. Testing')
    plt.xlabel('Intensidad (0 = Negro absoluto, 255 = Blanco absoluto)')
    plt.ylabel('Frecuencia Relativa')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Ejecución:
analizar_intensidades(extract_dir)


# %% [markdown]
### Train y Test
train_dir = os.path.join(extract_dir, 'Training')
test_dir = os.path.join(extract_dir, 'Testing')


## Definimos los parámetros comunes
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

## Cargamos el set de ENTRENAMIENTO (80%)
## Cargamos el set de ENTRENAMIENTO (80% de la carpeta Training)
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

## Cargamos el set de VALIDACIÓN (20% de la carpeta Training)
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

## Cargamos el set de TEST (Carpeta Testing completa)
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
## Visualización de 5 imágenes

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1): # Tomamos un batch del dataset de entrenamiento
    for i in range(5): # Iteramos sobre las primeras 5 imágenes del batch
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8")) # Mostramos la imagen
        plt.title(train_ds.class_names[labels[i]]) # Establecemos el título con el nombre de la clase
        plt.axis("off") # Ocultamos los ejes
plt.show()


# %% [markdown]
### Preprocesamiento de Datos con capas dentro del modelo (Data Augmentatio)
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

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# %% [markdown]
# Creación del modelo base (MobileNetV2)

# Cargar el modelo pre-entrenado MobileNetV2 sin las capas superiores
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                               include_top=False,
                                               weights='imagenet')

# Congelar el modelo base (importante para no destruir los pesos pre-entrenados)
base_model.trainable = False

#  Crea la arquitectura completa usando la API Funcional o Sequential
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  # Aplicamos tus capas de aumento
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # Normalización [-1, 1]
x = base_model(x, training=False) # Pasamos por MobileNetV2
x = tf.keras.layers.GlobalAveragePooling2D()(x) # Colapsamos las dimensiones espaciales
x = tf.keras.layers.Dropout(0.2)(x) # Regularización para evitar sobreajuste
outputs = tf.keras.layers.Dense(4, activation='softmax')(x) # Capa final (4 tumores)

model = tf.keras.Model(inputs, outputs)

# Complilación del modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary() # Para verificar la cantidad de parámetros

epochs = 20

# Callback para detenerse si el val_loss no mejora en 3 épocas
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping]
)
# %%
# Evaluar en el set de test
loss, accuracy = model.evaluate(test_ds)
print(f"Precisión en Test: {accuracy*100:.2f}%")

# Reporte de clasificación (opcional pero muy recomendado)
from sklearn.metrics import classification_report

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_ds), axis=-1)

# 1. Creas el dataset
# 1. Definimos las clases primero (esto es lo que usará el classification_report al final)
MIS_CLASES_REALES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 2. Carga de Entrenamiento y Validación (Usando la variable train_dir que ya tienes)
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    class_names=MIS_CLASES_REALES,
    subset="training",
    seed=42,
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    class_names=MIS_CLASES_REALES,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 3. Carga de Test (Usando test_dir)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    class_names=MIS_CLASES_REALES,
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 4. Optimizamos (Aquí es donde el objeto cambia de tipo, pero ya guardamos las clases arriba)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
# %%

# Usa la variable que definimos al principio del proceso de carga
print(classification_report(y_true, y_pred, target_names=MIS_CLASES_REALES))