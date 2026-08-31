import pandas as pd
import numpy as np

# Configuración de la generación sintética
np.random.seed(42)
n_registros_por_persona = 1000  # 1000 muestras por cada persona
personas = ["Carlos_Molina", "Ana_Perez", "Luis_Gomez"]

datos = []
for persona in personas:
    # Definimos tendencias ligeramente distintas para que el modelo aprenda a diferenciarlos
    offset_area = np.random.uniform(8000, 25000)
    offset_aspect = np.random.uniform(0.6, 2.2)
    offset_ang = np.random.uniform(-15, 15)
    
    for _ in range(n_registros_por_persona):
        area_tinta = max(500, np.random.normal(loc=offset_area, scale=2500))
        aspect_ratio = max(0.1, np.random.normal(loc=offset_aspect, scale=0.2))
        inclinacion_grados = np.random.normal(loc=offset_ang, scale=4.0)
        
        datos.append({
            'area_tinta': round(float(area_tinta), 4),
            'aspect_ratio': round(float(aspect_ratio), 4),
            'inclinacion_grados': round(float(inclinacion_grados), 4),
            'persona': persona
        })

# Crear DataFrame y exportar a CSV
df = pd.DataFrame(datos)
df.to_csv("dataset_caligrafia.csv", index=False)

print("¡Éxito! Se han generado 3000 registros en el archivo 'dataset_caligrafia.csv'.")