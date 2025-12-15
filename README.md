🎤 Proyecto Final de Reconocimiento de Voz

Este proyecto implementa un sistema de reconocimiento de dígitos hablados utilizando redes neuronales recurrentes (GRU) en Python.

📥 Descarga del Dataset

Para ejecutar reconocimientoDeNumeros.py, necesitas el siguiente dataset:

👉 Audio MNIST Dataset:
https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist/data

💾 Archivos generados y guardados

Durante la fase de preprocesamiento, el sistema guarda automáticamente los siguientes archivos:

X_train.npy
y_train.npy
X_val.npy
y_val.npy
X_test.npy
y_test.npy
digit_recognition_gru_model.keras


🧠 Entrenamiento, Evaluación y Pruebas

Con los archivos .npy disponibles, puedes ejecutar todos los scripts del proyecto sin necesidad de reentrenar el modelo.

🔹 1. Entrenar el modelo
python train_model.py


Esto genera el modelo entrenado:

digit_recognition_gru_model.keras

🔹 2. Evaluar el modelo
python evaluate_model.py

🔹 3. Probar el modelo en vivo
python live_predict.py


Este archivo permite hacer predicciones en tiempo real usando el micrófono.

✔️ Notas finales

Todos los scripts están listos para ejecutarse.

El archivo digit_recognition_gru_model.keras producido por train_model.py permite compilar y ejecutar correctamente los demás scripts.

No es necesario rehacer el dataset si ya cuentas con los archivos .npy incluidos.
