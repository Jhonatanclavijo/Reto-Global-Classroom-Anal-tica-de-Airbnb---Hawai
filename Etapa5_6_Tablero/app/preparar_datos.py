"""
Prepara los datos que usa el tablero a partir de los archivos del repositorio.
Se ejecuta UNA vez (antes de construir la imagen Docker):

    python preparar_datos.py

Lee   ../../listings_clean.csv, ../../clasificacion_train.csv y
      ../../clasificacion_sin_evidencia.csv
Crea  data/anuncios.csv.gz   (datos para las visualizaciones + probabilidad
                              del modelo de Uniandes para cada anuncio)
      data/referencias.json  (centroides de barrios, listas de categorías y
                              valores por defecto del formulario)
"""
import json
import os

import pandas as pd

from modelos import POR_ID, AMENIDADES

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, "..", ".."))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

limpio = pd.read_csv(os.path.join(REPO, "listings_clean.csv"))
train = pd.read_csv(os.path.join(REPO, "clasificacion_train.csv"))
sin_ev = pd.read_csv(os.path.join(REPO, "clasificacion_sin_evidencia.csv"))
feats = pd.concat([train, sin_ev], ignore_index=True)

# Probabilidad del modelo de Uniandes para todos los anuncios
m = POR_ID["uniandes_red_clasificacion"]
feats["prob_uniandes"] = m.predecir(feats)

cols_viz = [
    "id", "price", "island", "neighbourhood", "latitude", "longitude",
    "room_type", "property_group", "accommodates", "bedrooms", "bathrooms",
    "review_scores_rating", "number_of_reviews", "n_amenities",
    "host_is_superhost", *AMENIDADES.keys(),
]
anuncios = limpio[cols_viz].merge(
    feats[["id", "nivel", "recommended", "prob_uniandes"]], on="id", how="left"
)
anuncios["prob_uniandes"] = anuncios["prob_uniandes"].round(4)
anuncios.to_csv(os.path.join(DATA, "anuncios.csv.gz"), index=False, compression="gzip")

med = train.median(numeric_only=True)
ref = {
    "islas": sorted(limpio["island"].unique().tolist()),
    "barrios": sorted(limpio["neighbourhood"].unique().tolist()),
    "isla_de_barrio": limpio.groupby("neighbourhood")["island"].first().to_dict(),
    "centroides": {
        b: [float(g["latitude"].median()), float(g["longitude"].median())]
        for b, g in limpio.groupby("neighbourhood")
    },
    "room_types": sorted(limpio["room_type"].unique().tolist()),
    "property_groups": sorted(limpio["property_group"].unique().tolist()),
    "host_location_groups": sorted(limpio["host_location_group"].unique().tolist()),
    "tasa_global_recomendados": float(train["recommended"].mean()),
    "medianas": {
        k: float(med[k]) for k in [
            "availability_30", "availability_365", "maximum_nights",
            "minimum_nights", "n_amenities",
        ]
    },
}
with open(os.path.join(DATA, "referencias.json"), "w", encoding="utf-8") as fh:
    json.dump(ref, fh, ensure_ascii=False, indent=1)

print(f"anuncios: {len(anuncios)} filas · referencias guardadas en {DATA}")
