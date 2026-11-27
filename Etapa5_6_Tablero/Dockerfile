FROM python:3.12-slim

# Usuario sin privilegios que ejecuta el tablero
RUN adduser --disabled-password --gecos '' dash-user
WORKDIR /opt/app

# Dependencias primero (se aprovecha la caché de Docker)
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Código, datos y modelos serializados (se cargan pre-entrenados al iniciar)
COPY app/ .
RUN chown -R dash-user:dash-user /opt/app

USER dash-user
ENV TF_CPP_MIN_LOG_LEVEL=2
EXPOSE 8050

# 1 worker (TensorFlow ocupa memoria) con varios hilos
CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050", "--workers", "1", "--threads", "4", "--timeout", "120"]
