# Etapas 5 y 6: tablero en Dash y despliegue en AWS con Docker

Tablero para **viajeros** que buscan alojamiento de Airbnb en Hawái. Tiene tres pestañas:

1. **Buscar alojamiento**: filtros de isla, tipo de alojamiento, presupuesto y huéspedes, indicadores
   y tres visualizaciones: (1) un mapa de anuncios coloreado por la probabilidad de ser recomendado
   o por el precio, (2) el precio por zona en diagramas de caja y (3) la diferencia en % de
   recomendados (o en precio) según cada amenidad. Al final hay una tabla con los 25 alojamientos
   sugeridos, que incluye anuncios nuevos cuya recomendación estima el modelo.
2. **Evaluar un anuncio**: un selector de modelo y un formulario con los datos de un anuncio. El
   resultado muestra la probabilidad de que el anuncio sea recomendado, si su precio es razonable
   frente a anuncios similares y qué amenidades le faltan frente a los recomendados de la isla.
3. **Instrucciones**: cómo usar el tablero, qué significa "recomendado" y qué modelos hay.

Los modelos se cargan **pre-entrenados desde archivos locales** (`app/modelos/`) al iniciar.

## Estructura

```
Etapa5_6_Tablero/
├── Dockerfile
├── README.md
└── app/
    ├── app.py               # tablero Dash (layout + callbacks)
    ├── modelos.py           # carga de modelos y selector (aquí se agrega el del Tec)
    ├── preparar_datos.py    # genera data/ a partir de los CSV del repositorio (se corre una vez)
    ├── requirements.txt
    ├── assets/style.css
    ├── data/                # anuncios.csv.gz y referencias.json (datos de las gráficas)
    └── modelos/uniandes/    # copia de Modelo_final (Keras + escalador + configuración)
```

## Correr en local (sin Docker)

```bash
cd Etapa5_6_Tablero/app
pip install -r requirements.txt
python app.py            # abrir http://localhost:8050
```

`data/` ya viene generada. Solo hay que volver a correr `python preparar_datos.py` si cambian
`listings_clean.csv`, los archivos `clasificacion_*.csv` o el modelo de Uniandes.

> **Importante:** el escalador (`scaler.joblib`) se guardó con **scikit-learn 1.7.2**. Por eso
> `requirements.txt` fija esa versión. Si el modelo se reentrena con otra versión, hay que actualizarla.

## Correr con Docker (local)

```bash
cd Etapa5_6_Tablero
docker build -t tablero-airbnb-hawai .
docker run -d --name tablero -p 8050:8050 --restart unless-stopped tablero-airbnb-hawai
# abrir http://localhost:8050
```

La imagen ocupa cerca de 2,7 GB (por TensorFlow) y el contenedor usa unos 400 MB de RAM.

## Despliegue en AWS (EC2)

1. **Crear la instancia** (consola de AWS, servicio EC2, *Launch instance*):
   - AMI: Ubuntu Server 24.04 LTS.
   - Tipo: `t3.small` (2 GB de RAM). `t2.micro`/`t3.micro` (1 GB) puede funcionar, pero hay
     que agregar swap (paso 3).
   - Almacenamiento: **20 GB** (con los 8 GB por defecto no alcanza para construir la imagen).
   - *Security group*: permitir SSH (22) desde su IP y **TCP 8050 desde 0.0.0.0/0**.
   - Crear o usar un *key pair* (`.pem`).
2. **Conectarse**: `ssh -i llave.pem ubuntu@<IP_PUBLICA>`
3. **Instalar Docker y Git** (y swap si la instancia es de 1 GB):
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io git
   sudo usermod -aG docker ubuntu && newgrp docker
   # solo para instancias de 1 GB:
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
   ```
4. **Clonar el repositorio y construir la imagen**:
   ```bash
   git clone https://github.com/Jhonatanclavijo/Reto-Global-Classroom-Anal-tica-de-Airbnb---Hawai.git
   cd Reto-Global-Classroom-Anal-tica-de-Airbnb---Hawai/Etapa5_6_Tablero
   docker build -t tablero-airbnb-hawai .
   ```
5. **Lanzar el contenedor** (se reinicia solo si la máquina se reinicia):
   ```bash
   docker run -d --name tablero -p 8050:8050 --restart unless-stopped tablero-airbnb-hawai
   docker ps            # verificar que está arriba
   docker logs tablero  # ver el arranque de gunicorn
   ```
6. **Abrir el tablero**: `http://<IP_PUBLICA>:8050`

### Pantallazos que pide el reporte de la Etapa 6

- La consola de EC2 con la instancia en *Running*, donde se vean la **IP pública**, la **IP
  privada** y el **usuario de AWS** (esquina superior derecha).
- La terminal con `docker build`, `docker run` y `docker ps`.
- En la misma terminal, `hostname -I` (IP privada) y `curl -s ifconfig.me` (IP pública).
- El navegador con el tablero abierto en `http://<IP_PUBLICA>:8050`.

### Mantenimiento

- Actualizar después de cambios en el repo:
  ```bash
  git pull && docker build -t tablero-airbnb-hawai . && docker rm -f tablero && docker run -d --name tablero -p 8050:8050 --restart unless-stopped tablero-airbnb-hawai
  ```
- Detener la instancia cuando no se use. Al detenerla, la IP pública cambia, salvo que se asocie
  una *Elastic IP*. Para el 3 de diciembre debe quedar encendida.

## Agregar los modelos del Tec de Monterrey

El selector ya muestra dos entradas del Tec como "(pendiente)". Se activan solas cuando existen sus archivos:

| Modelo | Archivos esperados |
|---|---|
| Regresión de precio | `app/modelos/tec_regresion/modelo.joblib` y `configuracion.json` (crear la carpeta) |
| Clasificación | `app/modelos/tec_clasificacion/modelo.joblib` y `configuracion.json` |

`configuracion.json` debe tener `{"columnas": [...], "umbral": 0.5, "salida": "log_price"}`. Las
`columnas` usan los mismos nombres de `listings_model.csv` / `clasificacion_train.csv`. `salida`
puede ser `"log_price"` o `"price"` para la regresión. Si el modelo necesita escalado, se pone
`scaler.joblib` en la misma carpeta. Si lo entregan en otro formato (statsmodels, pickle, R), solo
hay que ajustar su entrada en `REGISTRO` dentro de `app/modelos.py`.
