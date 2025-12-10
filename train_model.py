import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# --- Parámetros de Extracción (Asegúrate que coincidan con el código anterior) ---
N_MELS = 64
MAX_PAD_LEN = 50
NUM_CLASSES = 10 # Dígitos 0-9
# --------------------------------------------------------------------------------

def load_saved_data():
    """Carga los arrays de NumPy previamente guardados."""
    print("Cargando datos guardados...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')
    # X_test = np.load('X_test.npy') # Los usaremos más tarde para la evaluación final
    # y_test = np.load('y_test.npy')
    
    # Keras espera las etiquetas en formato 'one-hot encoding' para la Entropía Cruzada Categórica
    y_train_one_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_one_hot = to_categorical(y_val, num_classes=NUM_CLASSES)
    
    return X_train, y_train_one_hot, X_val, y_val_one_hot


def create_gru_model(input_shape):
    """
    Define y compila el modelo GRU para clasificación de dígitos.
    
    Arquitectura elegida:
    1. GRU (128 unidades) para capturar la secuencia temporal del audio.
    2. Dropout para prevenir el overfitting.
    3. Capa Densa (Softmax) para la clasificación final de 10 dígitos.
    """
    
    model = Sequential()
    
    # 1. Adaptar la entrada 2D (64, 50, 1) a la GRU.
    # Keras espera (secuencia_largo, características_por_paso). 
    # Nuestra entrada es (MAX_PAD_LEN, N_MELS) o (50, 64) si lo trasponemos.
    # Como la GRU maneja secuencias, debemos asegurarnos de que el primer eje
    # después de la muestra sea el eje temporal (frames).
    # La forma de entrada de la GRU será (MAX_PAD_LEN, N_MELS * 1)
    
    # Usamos Reshape para aplanar el eje de Mels y dejar la secuencia temporal 
    # en el primer eje. La entrada esperada es (50, 64)
    model.add(Reshape((MAX_PAD_LEN, N_MELS), input_shape=input_shape))
    
    # 2. Capa GRU
    model.add(GRU(
        units=128,          # 128 unidades GRU
        return_sequences=False # Solo necesitamos la salida del último paso
    ))
    # 
    
    # 3. Dropout (regularización)
    model.add(Dropout(0.5))
    
    # 4. Capa de Salida (10 Neuronas, Softmax)
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # 5. Compilación (Función de Pérdida: Entropía Cruzada Categórica)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model


def create_cnn_gru(input_shape):
    model = Sequential()

    # Entrada: (64, 50, 1)
    model.add(Reshape((N_MELS, MAX_PAD_LEN, 1), input_shape=input_shape))  # (64,50,1)

    # --- Bloque Convolucional ---
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))

    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))

    # Convertir a secuencia para GRU
    model.add(Reshape((-1, 64)))  # Ahora queda como secuencia para GRU

    # --- GRU ---
    model.add(GRU(128))
    model.add(Dropout(0.4))

    # Salida
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Entrena el modelo usando Early Stopping."""
    
    # Configuración de Early Stopping (Parada Temprana)
    # Monitorea la pérdida de validación y detiene el entrenamiento si no mejora
    # durante 10 épocas para prevenir el overfitting.
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    print("\nIniciando entrenamiento del modelo...")
    
    history = model.fit(
        X_train, y_train,
        epochs=50,             # Número máximo de épocas (se detendrá antes por Early Stopping)
        batch_size=32,          # Tamaño del lote
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    
    # Cargar datos
    X_train, y_train_oh, X_val, y_val_oh = load_saved_data()
    
    # Definir la forma de entrada para el modelo (64, 50, 1)
    input_shape = X_train.shape[1:]
    
    # Crear y compilar el modelo
    model = create_cnn_gru(input_shape)
    
    # Entrenar el modelo
    trained_model, history = train_model(model, X_train, y_train_oh, X_val, y_val_oh)
    
    # Guardar el modelo entrenado
    model_save_path = "digit_recognition_gru_model.keras"
    trained_model.save(model_save_path)
    print(f"\nModelo entrenado y guardado en: {model_save_path}")