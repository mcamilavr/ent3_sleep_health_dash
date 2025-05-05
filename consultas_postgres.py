import pandas as pd
from sqlalchemy import create_engine

# Conexión a PostgreSQL
username = 'postgres'
password = 'molly5011'
host = 'localhost'
port = '5432'
dbname = 'sleepdb'

# Crear motor de conexión
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{dbname}')

# Consulta 1: Mostrar las primeras filas
query1 = "SELECT * FROM sleep_data LIMIT 10;"
df1 = pd.read_sql(query1, engine)
print("\n🛏️ Primeras filas del dataset:")
print(df1)

# Consulta 2: Contar cuántos registros hay por tipo de trastorno
query2 = """
SELECT "Sleep Disorder", COUNT(*) AS cantidad
FROM sleep_data
GROUP BY "Sleep Disorder"
ORDER BY cantidad DESC;
"""
df2 = pd.read_sql(query2, engine)
print("\n📊 Cantidad de casos por tipo de trastorno del sueño:")
print(df2)

# Consulta 3: Promedio de duración del sueño según trastorno
query3 = """
SELECT "Sleep Disorder", AVG("Sleep Duration") AS promedio_duracion
FROM sleep_data
GROUP BY "Sleep Disorder";
"""
df3 = pd.read_sql(query3, engine)
print("\n😴 Promedio de horas de sueño según trastorno:")
print(df3)
