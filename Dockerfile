# Usa una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos al contenedor
COPY . /app

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto donde corre Dash
EXPOSE 8050

# Comando para ejecutar la aplicación
CMD ["python", "dashboard.py"]
