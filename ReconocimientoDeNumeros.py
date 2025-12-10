import numpy as np
import librosa
import os
import glob
from sklearn.model_selection import train_test_split

# --- Parámetros Globales (DEBEN COINCIDIR con la función extract_features anterior) ---
SR = 16000           # Frecuencia de muestreo
N_MELS = 64          # Número de filtros Mel
MAX_PAD_LEN = 50     # Longitud máxima de frames de tiempo
# -----------------------------------------------------------------------------------

def extract_features(file_path):
    """
    Función de extracción de características (Espectrograma Log-Mel)
    tal como se definió antes.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SR, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        current_len = log_mel_spectrogram.shape[1]
        
        if current_len > MAX_PAD_LEN:
            padded_spectrogram = log_mel_spectrogram[:, :MAX_PAD_LEN]
        
        elif current_len < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - current_len
            padded_spectrogram = np.pad(
                log_mel_spectrogram, 
                pad_width=((0, 0), (0, pad_width)), 
                mode='constant', 
                constant_values=-100.0
            )
        else:
            padded_spectrogram = log_mel_spectrogram
            
        # Agregamos una dimensión (canales) para que sea compatible con modelos Keras/TF (64, 50, 1)
        return np.expand_dims(padded_spectrogram, axis=-1)

    except Exception as e:
        # Esto captura errores de archivos corruptos o ilegibles
        print(f"Error procesando {file_path}: {e}")
        return None


def load_data(data_path="archive/data"):
    """
    Recorre el dataset, extrae características y organiza etiquetas.

    Args:
        data_path (str): Ruta a la carpeta que contiene los subdirectorios 01, 02, etc.

    Returns:
        tuple: (features_list, labels_list)
    """
    all_features = []
    all_labels = []
    total_files = 0
    
    # 1. Recorrer las carpetas (IDs de Hablantes: '01' a '60')
    # Usamos glob para encontrar todos los archivos .wav de forma recursiva.
    # El patrón '**/*.wav' busca en todos los subdirectorios de 'data_path'.
    file_paths = glob.glob(os.path.join(data_path, '**', '*.wav'), recursive=True)
    
    total_files = len(file_paths)
    print(f"Se encontraron {total_files} archivos .wav para procesar.")

    # 2. Iterar sobre las rutas de archivos
    for i, file_path in enumerate(file_paths):
        # Mostrar progreso
        if (i + 1) % 1000 == 0:
            print(f"-> Procesando archivo {i + 1}/{total_files}...")
            
        # Extraer el DÍGITO (la etiqueta) del nombre del archivo
        # El formato es: [dígito]_[hablante]_n.wav
        # Ejemplo: 'archive/data/01/1_01_0.wav' -> Etiqueta es '1'
        
        try:
            # Obtenemos solo el nombre del archivo (ej: '1_01_0.wav')
            file_name = os.path.basename(file_path)
            # El dígito es el primer caracter del nombre (antes del primer '_')
            label = int(file_name.split('_')[0])
            
            # 3. Aplicar Preprocesamiento y Extracción
            features = extract_features(file_path)
            
            if features is not None:
                all_features.append(features)
                all_labels.append(label)
        
        except ValueError:
            # Esto ignora archivos que no sigan el formato esperado
            print(f"Advertencia: Saltando archivo con formato de etiqueta incorrecto: {file_name}")
        except Exception as e:
            # Captura cualquier otro error durante la extracción
            print(f"Error crítico en {file_path}: {e}")
            
    print("\nProcesamiento completado.")
    return np.array(all_features), np.array(all_labels)


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    # La ruta al dataset es 'archive/data' si tu script está en la raíz 
    # donde se descomprimió el archivo 'audio-mnist'.
    DATA_ROOT = "archive/data" 
    
    print(f"Iniciando carga de datos desde: {DATA_ROOT}")
    
    # Cargar y extraer todos los datos
    X, y = load_data(data_path=DATA_ROOT)
    
    print("-" * 50)
    print("Datos cargados exitosamente.")
    print(f"Forma de las características (X): {X.shape}") # Debería ser algo como (30000, 64, 50, 1)
    print(f"Forma de las etiquetas (y): {y.shape}")     # Debería ser algo como (30000,)
    print("-" * 50)
    
    # 4. División de Datos (Paso 4.1 de tu metodología)
    # Dividir en Entrenamiento, Validación y Prueba (60/20/20)
    
    # Primero: Separar 20% para Prueba (Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Segundo: Separar el 20% restante (que es 25% del 80% que queda) para Validación (Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print("\nDivisión de Datos (60%/20%/20%) completada:")
    print(f"Entrenamiento (X_train): {X_train.shape[0]} muestras")
    print(f"Validación (X_val): {X_val.shape[0]} muestras")
    print(f"Prueba (X_test): {X_test.shape[0]} muestras")
    
    # 5. GUARDAR los datos preprocesados
    # Es altamente recomendable guardar los arrays de NumPy para no tener que 
    # ejecutar la extracción de características de nuevo.
    print("\nGuardando arrays de NumPy...")
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    print("Archivos guardados correctamente.")