import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def transform_data(df, categorical_columns, numerical_columns):
    # Convertir categorías a códigos numéricos
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convertir a string antes de la codificación
        label_encoders[col] = le

    # Convertir la columna target 'fase' a códigos numéricos
    target_encoder = LabelEncoder()
    df['fase'] = target_encoder.fit_transform(df['fase'].astype(str))

    # Normalización de características numéricas
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, label_encoders, target_encoder, scaler