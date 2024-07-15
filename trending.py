import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.io as pio
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Cargar los datos (ajusta la ruta al archivo según sea necesario)
df = pd.read_csv('SecopII_data_sample_small.csv', low_memory=False)

# Convertir la columna 'fecha_de_publicacion_del' a datetime
df['fecha_de_publicacion_del'] = pd.to_datetime(df['fecha_de_publicacion_del'])

# Cargar el modelo entrenado y el tokenizer
model = load_model('modelo_entrenado.h5')
tokenizer = joblib.load('tokenizer.pkl')

# Definir el valor máximo de longitud
MAXLEN = 500

# Preparar las columnas textuales para predicción
textual_data = df['descripci_n_del_procedimiento'].astype(str) + ' ' + df['nombre_del_procedimiento'].astype(str)
sequences = tokenizer.texts_to_sequences(textual_data)
X_text = pad_sequences(sequences, maxlen=MAXLEN)

# Hacer predicciones con el modelo CNN
y_pred = np.argmax(model.predict(X_text), axis=1)

# Agregar las predicciones al DataFrame
df['clasificacion'] = y_pred

# Contar el número de publicaciones por fecha y clasificación
publicaciones_por_fecha_clasificacion = df.groupby(['fecha_de_publicacion_del', 'clasificacion']).size().reset_index(name='num_publicaciones')

# Gráfico interactivo del número de publicaciones por fecha y clasificación
fig = px.line(publicaciones_por_fecha_clasificacion, x='fecha_de_publicacion_del', y='num_publicaciones', color='clasificacion', title='Número de Publicaciones por Clasificación a lo Largo del Tiempo')
pio.write_html(fig, file='numero_publicaciones_por_clasificacion.html')

# Crear una columna de mes y año
df['mes_anio'] = df['fecha_de_publicacion_del'].dt.to_period('M')

# Contar el número de publicaciones por mes, año y clasificación
publicaciones_mensuales_clasificacion = df.groupby(['mes_anio', 'clasificacion']).size().reset_index(name='num_publicaciones')

# Convertir 'mes_anio' a string para evitar problemas de serialización
publicaciones_mensuales_clasificacion['mes_anio'] = publicaciones_mensuales_clasificacion['mes_anio'].astype(str)

# Gráfico interactivo de tendencias mensuales por clasificación
fig = px.line(publicaciones_mensuales_clasificacion, x='mes_anio', y='num_publicaciones', color='clasificacion', title='Tendencias Mensuales de Publicaciones por Clasificación')
pio.write_html(fig, file='tendencias_mensuales_por_clasificacion.html')

# Calcular medias móviles
publicaciones_mensuales_clasificacion['MA3'] = publicaciones_mensuales_clasificacion.groupby('clasificacion')['num_publicaciones'].transform(lambda x: x.rolling(window=3).mean())
publicaciones_mensuales_clasificacion['MA6'] = publicaciones_mensuales_clasificacion.groupby('clasificacion')['num_publicaciones'].transform(lambda x: x.rolling(window=6).mean())

# Gráfico interactivo de tendencias mensuales con medias móviles por clasificación
fig = px.line(publicaciones_mensuales_clasificacion, x='mes_anio', y='num_publicaciones', color='clasificacion', line_group='clasificacion', title='Tendencias Mensuales con Medias Móviles por Clasificación')
fig.add_traces(px.line(publicaciones_mensuales_clasificacion, x='mes_anio', y='MA3', color='clasificacion', line_group='clasificacion').data)
fig.add_traces(px.line(publicaciones_mensuales_clasificacion, x='mes_anio', y='MA6', color='clasificacion', line_group='clasificacion').data)
pio.write_html(fig, file='tendencias_mensuales_con_medias_moviles.html')
