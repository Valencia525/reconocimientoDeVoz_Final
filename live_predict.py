import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import sounddevice as sd
import scipy.io.wavfile as wavfile
import os

# === CONFIGURACIÓN DEL MICRÓFONO ===
sd.default.samplerate = 16000   # Fuerza la tasa de muestreo
sd.default.channels = 1

# --- Parámetros GLOBALES (DEBEN COINCIDIR con el entrenamiento) ---
N_MELS = 64
MAX_PAD_LEN = 50
SR = 16000  # Frecuencia de muestreo usada en el modelo
MODEL_PATH = "digit_recognition_gru_model.keras"

# --- Parámetros de Grabación ---
DURATION = 1.0   # Grabar 1 segundos
# ----------------------------------------------------


def extract_features_for_inference(audio_data):
    """
    Procesa el audio grabado y devuelve el Log-Mel Spectrogram
    listo para el modelo.
    """
    try:
        # 1. Ya sabemos que el audio viene a 16k porque lo forzamos
        audio_resampled = librosa.resample(
            audio_data.astype(float),
            orig_sr=sd.default.samplerate,
            target_sr=SR
        )

        # 2. Espectrograma Mel
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_resampled,
            sr=SR,
            n_mels=N_MELS
        )

        # 3. Conversión a Log
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # 4. Padding o truncado
        current_len = log_mel_spectrogram.shape[1]

        if current_len > MAX_PAD_LEN:
            padded_spectrogram = log_mel_spectrogram[:, :MAX_PAD_LEN]
        elif current_len < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - current_len
            padded_spectrogram = np.pad(
                log_mel_spectrogram,
                pad_width=((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=-100.0
            )
        else:
            padded_spectrogram = log_mel_spectrogram

        # 5. Ajustar dimensiones para el modelo
        features = np.expand_dims(padded_spectrogram, axis=-1)
        features = np.expand_dims(features, axis=0)

        return features

    except Exception as e:
        print(f"Error procesando el audio: {e}")
        return None


def record_and_predict(model):
    """Graba audio, extrae características y predice el dígito."""
    print("-" * 50)
    print(f"Grabando {DURATION} segundos...")
    print(">>> Hable el DÍGITO (0–9) AHORA...")

    try:
        # Grabar audio
        samples = int(DURATION * sd.default.samplerate)
        recording = sd.rec(samples, samplerate=sd.default.samplerate,
                           channels=1, dtype='float64')
        sd.wait()

        print("Grabación finalizada. Procesando...")

        # Extraer características
        features = extract_features_for_inference(recording.flatten())

        if features is None:
            print("No se pudieron extraer características.")
            return

        # Hacer predicción
        predictions = model.predict(features, verbose=0)
        predicted_digit = np.argmax(predictions)
        confidence = predictions[0][predicted_digit] * 100

        print("-" * 50)
        print(f"RESULTADO: Dígito predicho → {predicted_digit}")
        print(f"Confianza: {confidence:.2f}%")
        print("-" * 50)

    except Exception as e:
        print(f"ERROR durante la grabación o predicción: {e}")
        print("Revisa que el micrófono esté conectado y accesible.")


# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    try:
        model = load_model(MODEL_PATH)
        print(f"Modelo cargado correctamente desde: {MODEL_PATH}")

        while True:
            record_and_predict(model)
            if input("¿Quieres probar otro dígito? (s/n): ").lower() != "s":
                break

    except Exception as e:
        print(f"ERROR al cargar el modelo: {e}")
