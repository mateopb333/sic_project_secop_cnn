import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

def prepare_data_for_cnn(df, textual_columns, numerical_columns):
    # Configuración del logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Verificación de las columnas
    if not all(col in df.columns for col in textual_columns):
        logging.error("Algunas columnas textuales no se encuentran en el DataFrame.")
        return None, None, None, None, None
    if not all(col in df.columns for col in numerical_columns):
        logging.error("Algunas columnas numéricas no se encuentran en el DataFrame.")
        return None, None, None, None, None
    
    # Tokenización y Padding de Texto
    logging.info("Tokenizando y aplicando padding a las columnas textuales.")
    text_data = df[textual_columns].astype(str).apply(lambda x: ' '.join(x), axis=1)
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text_data)
    
    # Convertir texto a secuencias
    sequences = tokenizer.texts_to_sequences(text_data)
    
    # Padding de las secuencias
    maxlen = 200  # Longitud máxima de la secuencia
    X_text = pad_sequences(sequences, maxlen=maxlen)
    
    # Crear embeddings con tensorflow
    vocab_size = min(len(tokenizer.word_index) + 1, 5000)  # Limitar el vocabulario a 5000 palabras
    embedding_dim = 100
    
    # Normalización de las características numéricas
    logging.info("Normalizando las columnas numéricas.")
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(df[numerical_columns].values)
    
    # Concatenar características numéricas y textuales
    X = np.hstack([X_numeric, X_text])
    
    # Preparar las etiquetas (target)
    y = df['fase'].values
    
    # Aplicar SMOTE para balancear las clases
    logging.info("Aplicando SMOTE para balancear las clases.")
    smote = SMOTE(random_state=42, k_neighbors=1)
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info("SMOTE aplicado exitosamente.")
    except ValueError as e:
        logging.error(f"Error de SMOTE: {e}")
        X_resampled, y_resampled = X, y  # Usar los datos originales si SMOTE falla
    
    return X_resampled, y_resampled, tokenizer, vocab_size, embedding_dim
