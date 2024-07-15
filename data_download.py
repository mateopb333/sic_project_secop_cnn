import requests
import pandas as pd

def get_socrata_data(base_url, limit, offset):
    url = f"{base_url}?$limit={limit}&$offset={offset}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error al obtener los datos:", response.status_code)
        return []

def download_data(base_url, limit, total_records, ruta_destino):
    data_frames = []
    for offset in range(0, total_records, limit):
        data = get_socrata_data(base_url, limit, offset)
        if data:
            df = pd.DataFrame(data)
            data_frames.append(df)
        else:
            break
    full_data = pd.concat(data_frames, ignore_index=True)
    full_data.to_csv(ruta_destino, encoding='utf-8', index=False)
    return full_data







