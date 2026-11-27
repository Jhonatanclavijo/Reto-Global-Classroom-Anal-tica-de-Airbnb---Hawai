"""
Registro de modelos del tablero.

Cada modelo se carga PRE-ENTRENADO desde archivos locales (carpeta ./modelos),
como lo pide la Etapa 6. Para agregar un modelo nuevo (por ejemplo los del
Tecnológico de Monterrey) basta con:

  1. Copiar sus archivos a ./modelos/<carpeta>/
  2. Agregar una entrada en REGISTRO (al final de este archivo).

Todos los modelos reciben el mismo diccionario de características construido
por `construir_caracteristicas()` (mismos nombres de columnas que
clasificacion_train.csv / listings_model.csv), y cada uno toma de ahí las
columnas que necesita según su archivo de configuración.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_MODELOS = os.path.join(BASE, "modelos")

AMENIDADES = {
    "am_wifi": "Wifi",
    "am_kitchen": "Cocina",
    "am_air_conditioning": "Aire acondicionado",
    "am_free_parking": "Parqueadero gratis",
    "am_pool": "Piscina",
    "am_hot_tub": "Jacuzzi",
    "am_washer": "Lavadora",
    "am_dishwasher": "Lavavajillas",
    "am_tv": "TV",
    "am_bbq": "Parrilla / BBQ",
    "am_beach_waterfront": "Frente a la playa",
    "am_ocean_view": "Vista al mar",
    "am_gym": "Gimnasio",
    "am_elevator": "Ascensor",
    "am_self_checkin": "Check-in autónomo",
    "am_workspace": "Espacio de trabajo",
    "am_patio_balcony": "Patio o balcón",
}


def _slug(texto: str) -> str:
    return (
        texto.lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    )


def construir_caracteristicas(p: dict, ref: dict) -> dict:
    """Convierte las entradas amigables del formulario en las variables
    codificadas que usan los modelos (one-hot, log del precio, etc.).

    p   : entradas del usuario (ver app.py)
    ref : referencias precalculadas (centroides de barrios, medianas)
    """
    barrio = p["neighbourhood"]
    isla = ref["isla_de_barrio"][barrio]
    lat, lon = ref["centroides"][barrio]
    min_noches = int(p["minimum_nights"])

    f = {
        "log_price": float(np.log(max(float(p["price"]), 1.0))),
        "latitude": lat,
        "longitude": lon,
        "accommodates": int(p["accommodates"]),
        "bedrooms": int(p["bedrooms"]),
        "beds": int(p["beds"]),
        "bathrooms": float(p["bathrooms"]),
        "bath_shared": int(bool(p["bath_shared"])),
        "minimum_nights": min_noches,
        "maximum_nights": int(p["maximum_nights"]),
        "long_stay": int(min_noches >= 30),
        "availability_30": int(p["availability_30"]),
        "availability_365": int(p["availability_365"]),
        "host_has_profile_pic": int(bool(p["host_has_profile_pic"])),
        "host_identity_verified": int(bool(p["host_identity_verified"])),
        "host_years": float(p["host_years"]),
        "calculated_host_listings_count": int(p["host_listings"]),
        "has_license": int(bool(p["has_license"])),
        # Historial del anfitrión: tasa de anuncios recomendados en sus otros
        # anuncios. Un anfitrión nuevo toma la tasa global (igual que en el
        # entrenamiento, donde host_has_history = 0).
        "host_has_history": int(int(p["host_hist_n"]) > 0),
        "host_hist_rate": (
            float(p["host_hist_rate"]) if int(p["host_hist_n"]) > 0
            else ref["tasa_global_recomendados"]
        ),
        "host_hist_n": int(p["host_hist_n"]),
    }
    amen = set(p.get("amenities") or [])
    for col in AMENIDADES:
        f[col] = int(col in amen)
    # n_amenities cuenta TODAS las amenidades del anuncio (no solo las 17
    # anteriores). Se usa el valor que ingrese el usuario, con mínimo las marcadas.
    f["n_amenities"] = max(int(p["n_amenities"]), len(amen))

    for i in ref["islas"]:
        f[f"island_{_slug(i)}"] = int(i == isla)
    for b in ref["barrios"]:
        f[f"neighbourhood_{_slug(b)}"] = int(b == barrio)
    for r in ref["room_types"]:
        f[f"room_type_{_slug(r)}"] = int(r == p["room_type"])
    for g in ref["property_groups"]:
        f[f"property_group_{_slug(g)}"] = int(g == p["property_group"])
    for h in ref["host_location_groups"]:
        f[f"host_location_group_{_slug(h)}"] = int(h == p["host_location_group"])
    return f


@dataclass
class Modelo:
    id: str
    nombre: str
    equipo: str
    tipo: str  # "clasificacion" o "regresion"
    descripcion: str
    archivos: list[str]
    cargador: Callable[["Modelo"], None]
    columnas: list[str] = field(default_factory=list)
    umbral: float | None = None
    _predictor: Callable | None = None

    @property
    def disponible(self) -> bool:
        return all(os.path.exists(os.path.join(DIR_MODELOS, a)) for a in self.archivos)

    def cargar(self):
        if self._predictor is None and self.disponible:
            self.cargador(self)

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """Clasificación: probabilidad de 'recomendado'. Regresión: precio USD/noche."""
        self.cargar()
        if self._predictor is None:
            raise RuntimeError(f"El modelo {self.nombre} no está disponible.")
        return self._predictor(X[self.columnas])


# ---------------------------------------------------------------- cargadores
def _cargar_keras_uniandes(m: Modelo):
    import joblib
    import keras

    carpeta = os.path.join(DIR_MODELOS, "uniandes")
    with open(os.path.join(carpeta, "configuracion.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    m.columnas = cfg["columnas"]
    m.umbral = float(cfg["umbral"])
    escalador = joblib.load(os.path.join(carpeta, "scaler.joblib"))
    red = keras.models.load_model(
        os.path.join(carpeta, "modelo_clasificacion.keras"), compile=False
    )

    def predecir(X: pd.DataFrame) -> np.ndarray:
        Xs = escalador.transform(X.to_numpy(dtype="float32"))
        return red.predict(Xs, verbose=0).ravel()

    m._predictor = predecir


def _cargador_joblib(carpeta: str, archivo_modelo: str):
    """Cargador genérico para modelos scikit-learn / statsmodels guardados con
    joblib + un configuracion.json con {"columnas": [...], "umbral": opcional,
    "salida": "log_price" | "price" | "probabilidad"}."""

    def cargar(m: Modelo):
        import joblib

        ruta = os.path.join(DIR_MODELOS, carpeta)
        with open(os.path.join(ruta, "configuracion.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        m.columnas = cfg["columnas"]
        m.umbral = cfg.get("umbral")
        salida = cfg.get("salida", "probabilidad" if m.tipo == "clasificacion" else "price")
        modelo = joblib.load(os.path.join(ruta, archivo_modelo))
        escalador = None
        if os.path.exists(os.path.join(ruta, "scaler.joblib")):
            escalador = joblib.load(os.path.join(ruta, "scaler.joblib"))

        def predecir(X: pd.DataFrame) -> np.ndarray:
            Z = escalador.transform(X) if escalador is not None else X
            if m.tipo == "clasificacion":
                return modelo.predict_proba(Z)[:, 1]
            y = np.asarray(modelo.predict(Z)).ravel()
            return np.exp(y) if salida == "log_price" else y

        m._predictor = predecir

    return cargar


# ------------------------------------------------------------------ registro
REGISTRO: list[Modelo] = [
    Modelo(
        id="uniandes_red_clasificacion",
        nombre="Uniandes · Red neuronal de clasificación",
        equipo="Uniandes",
        tipo="clasificacion",
        descripcion=(
            "Red densa (Keras) de 3 capas de 57 neuronas, activación GELU. "
            "Estima la probabilidad de que el anuncio sea recomendado "
            "(≥5 reseñas, ≥1 en el último año, calificación ≥4.8 y valor ≥4.7)."
        ),
        archivos=[
            "uniandes/modelo_clasificacion.keras",
            "uniandes/scaler.joblib",
            "uniandes/configuracion.json",
        ],
        cargador=_cargar_keras_uniandes,
    ),
    # SUPUESTO: el Tec entrega un modelo de regresión de precio serializado con
    # joblib. Si lo entregan en otro formato (p. ej. statsmodels .pickle o .pkl)
    # solo hay que ajustar `archivos` y el cargador.
    Modelo(
        id="tec_regresion_precio",
        nombre="Tec de Monterrey · Regresión de precio",
        equipo="Tec de Monterrey",
        tipo="regresion",
        descripcion="Modelo estadístico que estima el precio por noche (USD).",
        archivos=["tec_regresion/modelo.joblib", "tec_regresion/configuracion.json"],
        cargador=_cargador_joblib("tec_regresion", "modelo.joblib"),
    ),
    Modelo(
        id="tec_clasificacion",
        nombre="Tec de Monterrey · Clasificación",
        equipo="Tec de Monterrey",
        tipo="clasificacion",
        descripcion="Modelo estadístico de clasificación recomendado / no recomendado.",
        archivos=["tec_clasificacion/modelo.joblib", "tec_clasificacion/configuracion.json"],
        cargador=_cargador_joblib("tec_clasificacion", "modelo.joblib"),
    ),
]

POR_ID = {m.id: m for m in REGISTRO}
