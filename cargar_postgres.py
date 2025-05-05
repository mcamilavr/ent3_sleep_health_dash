import pandas as pd
from sqlalchemy import create_engine

# Leer el dataset del proyecto actual
#df = pd.read_csv(r'C:\Users\mcvar\OneDrive\Documentos\visualizacion\actividad3_viz\Sleep_health_and_lifestyle_dataset.csv')
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')


# Parámetros de conexión a tu servidor PostgreSQL
username = 'postgres'         # Ajusta si tu usuario es diferente
password = 'molly5011'        # Tu contraseña
host = 'localhost'            # O la IP si no estás en local
port = '5432'                 # Puerto por defecto de PostgreSQL
dbname = 'sleepdb'            # El nombre que le vas a poner a la nueva base de datos

# Crear el motor de conexión SQLAlchemy
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{dbname}')

# Subir el DataFrame a PostgreSQL en una tabla llamada 'sleep_data'
df.to_sql('sleep_data', engine, index=False, if_exists='replace')

print("✅ ¡Datos cargados exitosamente en la tabla 'sleep_data' de la base 'sleepdb'!")
