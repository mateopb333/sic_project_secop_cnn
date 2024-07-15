import logging
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from model_evaluation import evaluate_model, predict_example

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    try:
        # Cargar el modelo guardado
        logging.info('Cargando el modelo guardado.')
        model = load_model('modelo_entrenado_secop.h5')
        logging.info('Modelo cargado.')

        # Cargar los datos procesados
        logging.info('Cargando los datos procesados.')
        data = np.load('datos_procesados.npz')
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        logging.info('Datos procesados cargados.')

        # Cargar el historial de entrenamiento
        histories = joblib.load('histories.pkl')
        logging.info('Historial de entrenamiento cargado.')

        # Evaluar el modelo
        logging.info('Evaluando el modelo cargado.')
        evaluate_model(model, histories, X_test, y_test)
        logging.info('Evaluación del modelo completada.')

        # Probar la función predict_example
        logging.info('Probando la función predict_example.')
        tokenizer = joblib.load('tokenizer.pkl')
        
        # Ejemplo de texto para predecir
        example_text = "Contrato para la construcción de un puente en la ciudad de Bogotá" 
        maxlen = 500  # Debe ser el mismo que se usó en el entrenamiento
        prediction = predict_example(model, tokenizer, example_text, maxlen)
        
        # Decodificar la predicción
        target_encoder = joblib.load('target_encoder.pkl')
        predicted_class = target_encoder.inverse_transform([np.argmax(prediction)])
        
        logging.info(f'Predicción para el texto de ejemplo: {predicted_class}')

    except Exception as e:
        logging.error(f"Error en la ejecución del script: {e}")

