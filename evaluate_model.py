import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parámetros ---
NUM_CLASSES = 10 # Dígitos 0-9
MODEL_PATH = "digit_recognition_gru_model.keras" # Ruta donde se guardó el modelo
# ------------------

def evaluate_final_model():
    """
    Carga el modelo y los datos de prueba para calcular las métricas finales
    y visualizar la matriz de confusión.
    """
    
    # 1. Cargar el modelo entrenado
    print(f"Cargando el modelo desde: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo. Asegúrate de que el archivo '{MODEL_PATH}' existe.")
        print(f"Detalle del error: {e}")
        return

    # 2. Cargar datos de prueba (Test)
    print("Cargando datos de prueba (X_test, y_test)...")
    try:
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy') # Etiquetas categóricas (0, 1, 2, ...)
    except FileNotFoundError:
        print("ERROR: No se encontraron los archivos X_test.npy o y_test.npy. Ejecuta primero el script de carga de datos.")
        return

    # 3. Predicciones
    print("Realizando predicciones en el conjunto de prueba...")
    # La predicción es un array de probabilidades (one-hot)
    y_pred_probs = model.predict(X_test)
    
    # Convertir las probabilidades a etiquetas de clase (el índice con la mayor probabilidad)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # 4. Cálculo de Métricas
    
    # Tasa de Aciertos (Accuracy)
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Reporte de Clasificación (incluye precisión, recall y F1-score por clase)
    class_names = [str(i) for i in range(NUM_CLASSES)]
    report = classification_report(y_test, y_pred_classes, target_names=class_names)

    # 5. Mostrar Resultados
    print("-" * 60)
    print("           RESULTADOS DE EVALUACIÓN FINAL         ")
    print("-" * 60)
    print(f"Tasa de Aciertos (Accuracy) Final: {accuracy:.4f}")
    print("\nReporte de Clasificación por Clase:")
    print(report)
    print("-" * 60)

    # 6. Visualizar Matriz de Confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True,          # Mostrar los números en cada celda
        fmt='d',             # Formato de números enteros
        cmap='Blues',        # Mapa de color
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Matriz de Confusión (Dígitos 0-9)')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción del Modelo')
    plt.show()
    # 

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    evaluate_final_model()
    