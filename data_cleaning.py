import pandas as pd
import logging

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Eliminar duplicados
    df.drop_duplicates(inplace=True)

    # Columnas relevantes para el análisis
    textual_columns = ['descripci_n_del_procedimiento', 'nombre_del_procedimiento']
    categorical_columns = [
        'entidad', 'departamento_entidad', 'ciudad_entidad', 'ordenentidad',
        'modalidad_de_contratacion', 'justificaci_n_modalidad_de', 'tipo_de_contrato'
    ]
    numerical_columns = [
        'precio_base', 'duracion', 'proveedores_invitados', 'proveedores_con_invitacion',
        'visualizaciones_del', 'proveedores_que_manifestaron', 'respuestas_al_procedimiento',
        'respuestas_externas', 'conteo_de_respuestas_a_ofertas', 'proveedores_unicos_con',
        'numero_de_lotes', 'valor_total_adjudicacion'
    ]

    # Manejo de valores nulos en columnas textuales
    df[textual_columns] = df[textual_columns].fillna('')

    # Manejo de valores nulos en columnas numéricas y categóricas
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)  # Imputación con mediana para columnas numéricas

    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Imputación con moda para columnas categóricas

    return df, textual_columns, categorical_columns, numerical_columns