import logging
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
from model_training import ModelTrainer
from model_evaluation import evaluate_model

# Importar las funciones de los módulos
from data_download import download_data
from data_cleaning import load_and_clean_data
from data_transformation import transform_data
from data_preparation import prepare_data_for_cnn
from model_construction import build_simple_model

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    try:
        # Parámetros para descarga de datos
        base_url = "https://www.datos.gov.co/resource/p6dx-8zbt.json"
        limit = 10000
        total_records = 200000
        ruta_destino = 'SecopII_data_sample_small.csv'

        # Descarga de datos (usando el conjunto de datos reducido)
        logging.info('Iniciando la descarga de datos.')
        full_data = download_data(base_url, limit, total_records, ruta_destino)
        logging.info('Descarga de datos completada.')

        # Verificar la cantidad de datos descargados
        logging.info(f'Datos descargados: {len(full_data)} registros')

        # Carga y limpieza de datos
        logging.info('Cargando y limpiando los datos.')
        df, textual_columns, categorical_columns, numerical_columns = load_and_clean_data(ruta_destino)
        logging.info('Carga y limpieza de datos completada.')

        # Verificar la cantidad de datos después de la limpieza
        logging.info(f'Datos después de la limpieza: {len(df)} registros')

        # Transformación de datos
        logging.info('Transformando los datos.')
        df, label_encoders, target_encoder, scaler = transform_data(df, categorical_columns, numerical_columns)
        logging.info('Transformación de datos completada.')

        # Guardar los transformadores
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump(target_encoder, 'target_encoder.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        logging.info('Transformadores guardados.')

        # Preparación de datos para CNN
        logging.info('Preparando los datos para el modelo CNN.')
        X, y, tokenizer, vocab_size, embedding_dim = prepare_data_for_cnn(df, textual_columns, numerical_columns)
        logging.info('Preparación de datos para CNN completada.')

        # Guardar el tokenizer
        joblib.dump(tokenizer, 'tokenizer.pkl')
        logging.info('Tokenizer guardado.')

        # Verificar la forma de los datos preparados
        logging.info(f'Forma de X: {X.shape}')
        logging.info(f'Forma de y: {y.shape}')

        # Dividir los datos en conjuntos de entrenamiento y prueba
        logging.info('Dividiendo los datos en conjuntos de entrenamiento y prueba.')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info('División de datos completada.')

        # Verificar la distribución de clases antes de aplicar SMOTE
        logging.info(f'Distribución de clases antes de SMOTE: {Counter(y_train)}')

        # Manejo de clases con muy pocas muestras antes de aplicar SMOTE
        min_samples_threshold = 2
        class_counts = Counter(y_train)
        classes_to_remove = [cls for cls, count in class_counts.items() if count < min_samples_threshold]
        for cls in classes_to_remove:
            indices_to_remove = np.where(y_train == cls)[0]
            X_train = np.delete(X_train, indices_to_remove, axis=0)
            y_train = np.delete(y_train, indices_to_remove, axis=0)
        logging.info(f'Removed classes with less than {min_samples_threshold} samples: {classes_to_remove}')

        # Aplicar SMOTE para balancear las clases
        logging.info('Aplicando SMOTE para balancear las clases.')
        smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logging.info('SMOTE aplicado.')

        # Ajustar etiquetas después de SMOTE para que estén dentro del rango correcto
        unique_classes = np.unique(y_train)
        target_encoder.fit(unique_classes)
        y_train = target_encoder.transform(y_train)
        
        # Ajustar y_test para que no tenga etiquetas no vistas
        y_test = np.array([label if label in unique_classes else -1 for label in y_test])
        valid_indices = y_test != -1
        X_test = X_test[valid_indices]
        y_test = y_test[valid_indices]
        y_test = target_encoder.transform(y_test)

        num_classes = len(unique_classes)
        logging.info(f'Número de clases: {num_classes}')

        # Guardar los datos de entrenamiento y prueba
        np.savez('datos_procesados.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        logging.info('Datos de entrenamiento y prueba guardados.')

        # Entrenamiento del modelo con K-Fold Cross-Validation
        logging.info('Iniciando el entrenamiento del modelo con K-Fold Cross-Validation.')
        trainer = ModelTrainer(build_simple_model, X_train, y_train, vocab_size, embedding_dim, input_length=X_train.shape[1], n_splits=5, epochs=20, batch_size=32)
        histories = trainer.train()
        logging.info('Entrenamiento del modelo con K-Fold Cross-Validation completado.')

        # Evaluación del modelo
        logging.info('Evaluando el modelo.')
        evaluate_model(trainer.model, histories, X_test, y_test)
        logging.info('Evaluación del modelo completada.')

    except Exception as e:
        logging.error(f"Error en la ejecución del script: {e}")
