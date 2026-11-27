"""
Tablero Airbnb Hawái · Reto Global Classroom (Uniandes + Tec de Monterrey)
Etapa 5: diseño y desarrollo del tablero en Dash.

Usuario final: el VIAJERO que busca alojamiento en Hawái. El tablero le ayuda a
encontrar anuncios recomendados según su presupuesto, zona y número de huéspedes,
incluso anuncios nuevos con pocas reseñas (el modelo estima si serían
recomendados), y a evaluar si un anuncio concreto es buena opción y buen precio.

Ejecutar en local:   python app.py            -> http://localhost:8050
En el contenedor:    gunicorn (ver Dockerfile)
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html

from modelos import AMENIDADES, POR_ID, REGISTRO, construir_caracteristicas

BASE = os.path.dirname(os.path.abspath(__file__))
DF = pd.read_csv(os.path.join(BASE, "data", "anuncios.csv.gz"))
with open(os.path.join(BASE, "data", "referencias.json"), encoding="utf-8") as fh:
    REF = json.load(fh)

# Los modelos disponibles se cargan desde archivos al iniciar (Etapa 6)
for _m in REGISTRO:
    _m.cargar()
# Umbral del modelo de Uniandes, usado para contar los anuncios nuevos recomendados
UMBRAL_MAPA = POR_ID["uniandes_red_clasificacion"].umbral or 0.5

TRAD_ROOM = {
    "Entire home/apt": "Vivienda completa",
    "Private room": "Habitación privada",
    "Hotel room": "Habitación de hotel",
    "Shared room": "Habitación compartida",
}
TRAD_PROP = {
    "Apartment": "Apartamento", "Condo": "Condominio", "Guest unit": "Unidad de huéspedes",
    "Hotel/Resort": "Hotel / resort", "House": "Casa", "Other": "Otro",
}
TRAD_ISLA = {"Hawaii": "Hawái (Big Island)", "Honolulu": "Oahu (Honolulu)",
             "Kauai": "Kauai", "Maui": "Maui"}

AZUL, CORAL, GRIS = "#1f6f8b", "#e4572e", "#8a94a6"
PLANTILLA = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#1d2433"),
    margin=dict(l=10, r=10, t=40, b=10),
)

app = Dash(__name__, title="Airbnb Hawái · Tablero", suppress_callback_exceptions=True)
server = app.server


# ------------------------------------------------------------------ helpers
def opciones(valores, trad=None):
    trad = trad or {}
    return [{"label": trad.get(v, v), "value": v} for v in valores]


def campo(etiqueta, componente, ayuda=None):
    hijos = [html.Label(etiqueta, className="campo-etiqueta"), componente]
    if ayuda:
        hijos.append(html.Div(ayuda, className="campo-ayuda"))
    return html.Div(hijos, className="campo")


def num(id_, valor, minimo, maximo, paso=1):
    return dcc.Input(id=id_, type="number", value=valor, min=minimo, max=maximo,
                     step=paso, className="entrada", debounce=True)


def kpi(titulo, valor, sub=""):
    return html.Div([html.Div(titulo, className="kpi-titulo"),
                     html.Div(valor, className="kpi-valor"),
                     html.Div(sub, className="kpi-sub")], className="kpi")


PRECIO_MAX = int(np.ceil(DF["price"].max() / 100) * 100)


def filtrar(islas, tipos, presupuesto=None, huespedes=None):
    d = DF
    if islas:
        d = d[d["island"].isin(islas)]
    if tipos:
        d = d[d["room_type"].isin(tipos)]
    if presupuesto:
        d = d[d["price"].between(*presupuesto)]
    if huespedes:
        d = d[d["accommodates"] >= huespedes]
    return d


FILTROS = [Input("f-isla", "value"), Input("f-tipo", "value"),
           Input("f-presupuesto", "value"), Input("f-huespedes", "value")]


# ------------------------------------------------------------------ layout
filtros = html.Div([
    campo("Isla", dcc.Dropdown(id="f-isla", options=opciones(REF["islas"], TRAD_ISLA),
                               multi=True, placeholder="Todas las islas")),
    campo("Tipo de alojamiento", dcc.Dropdown(id="f-tipo",
                                              options=opciones(REF["room_types"], TRAD_ROOM),
                                              multi=True, placeholder="Todos los tipos")),
    campo("Presupuesto por noche (USD)", dcc.RangeSlider(
        id="f-presupuesto", min=0, max=PRECIO_MAX, step=25, value=[0, PRECIO_MAX],
        marks={v: f"${v:,}" for v in range(0, PRECIO_MAX + 1, 1000)},
        tooltip={"placement": "bottom", "always_visible": False})),
    campo("Huéspedes (mínimo)", num("f-huespedes", 1, 1, 16)),
], className="filtros")

tab_explorar = html.Div([
    html.P("Encuentre dónde quedarse en Hawái. Indique su isla, tipo de alojamiento, "
           "presupuesto y número de huéspedes: las gráficas y la lista de sugeridos se "
           "actualizan con sus filtros.", className="intro"),
    filtros,
    html.Div(id="kpis", className="kpis"),
    html.Div([
        html.Div([
            html.Div([
                html.H3("1. ¿Dónde están los alojamientos recomendados?"),
                dcc.RadioItems(id="mapa-color", inline=True, value="prob_uniandes",
                               options=[{"label": " Probabilidad de ser recomendado",
                                         "value": "prob_uniandes"},
                                        {"label": " Precio por noche", "value": "price"}],
                               className="radio"),
            ], className="tarjeta-cabecera"),
            dcc.Graph(id="g-mapa", config={"displayModeBar": False}, style={"height": "460px"}),
        ], className="tarjeta ancho-completo"),
        html.Div([
            html.H3("2. ¿Cuánto cuesta cada zona?"),
            html.Div("Mitad central de los anuncios (caja) y mediana (línea). "
                     "Ordenado de mayor a menor mediana.", className="campo-ayuda"),
            dcc.Graph(id="g-precio", config={"displayModeBar": False}, style={"height": "460px"}),
        ], className="tarjeta"),
        html.Div([
            html.Div([
                html.H3("3. ¿Qué amenidades tienen los alojamientos mejor calificados?"),
                dcc.RadioItems(id="amen-metrica", inline=True, value="rec",
                               options=[{"label": " % recomendados", "value": "rec"},
                                        {"label": " Precio mediano", "value": "precio"}],
                               className="radio"),
            ], className="tarjeta-cabecera"),
            html.Div("Diferencia entre los anuncios que tienen la amenidad y los que no.",
                     className="campo-ayuda"),
            dcc.Graph(id="g-amen", config={"displayModeBar": False}, style={"height": "460px"}),
        ], className="tarjeta"),
        html.Div([
            html.H3("Alojamientos sugeridos"),
            html.Div("Los 25 anuncios con mayor probabilidad de ser recomendados dentro de sus "
                     "filtros (máximo 5 por zona). «Estimado por el modelo» indica anuncios nuevos o con pocas "
                     "reseñas, donde la recomendación la calcula el modelo de Uniandes.",
                     className="campo-ayuda"),
            dash_table.DataTable(
                id="t-sugeridos", page_size=25, sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "Inter, Segoe UI, sans-serif", "fontSize": 13,
                            "padding": "6px 10px", "textAlign": "left", "border": "none"},
                style_header={"fontWeight": 600, "backgroundColor": "#f5f7fa",
                              "borderBottom": "1px solid #e3e7ee"},
                style_data_conditional=[{"if": {"row_index": "odd"},
                                         "backgroundColor": "#fafbfc"}],
            ),
        ], className="tarjeta ancho-completo"),
    ], className="rejilla"),
])

med = REF["medianas"]
formulario = html.Div([
    html.H3("Datos del anuncio"),
    html.Div("Copie los datos que aparecen en la página del anuncio de Airbnb que le interesa.",
             className="campo-ayuda"),
    html.Div([
        campo("Isla", dcc.Dropdown(id="p-isla", options=opciones(REF["islas"], TRAD_ISLA),
                                   value="Maui", clearable=False)),
        campo("Barrio", dcc.Dropdown(id="p-barrio", clearable=False)),
        campo("Tipo de alojamiento", dcc.Dropdown(id="p-room", value="Entire home/apt",
                                                  options=opciones(REF["room_types"], TRAD_ROOM),
                                                  clearable=False)),
        campo("Tipo de propiedad", dcc.Dropdown(id="p-prop", value="Condo",
                                                options=opciones(REF["property_groups"], TRAD_PROP),
                                                clearable=False)),
        campo("Precio por noche (USD)", num("p-precio", 350, 20, 5000, 1),
              "El precio que muestra el anuncio."),
        campo("Huéspedes", num("p-huespedes", 4, 1, 16)),
        campo("Habitaciones", num("p-habitaciones", 2, 0, 10)),
        campo("Camas", num("p-camas", 2, 0, 16)),
        campo("Baños", num("p-banos", 1, 0, 10, 0.5)),
        campo("Baño compartido", dcc.Checklist(id="p-bano-comp", options=[{"label": " Sí", "value": 1}],
                                               value=[], className="check")),
    ], className="form-rejilla"),
    campo("Amenidades", dcc.Checklist(
        id="p-amen", options=[{"label": " " + v, "value": k} for k, v in AMENIDADES.items()],
        value=["am_wifi", "am_kitchen", "am_washer", "am_tv"], className="check-rejilla")),
    html.Details([
        html.Summary("Más datos del anuncio (anfitrión y disponibilidad, opcional)"),
        html.Div([
            campo("Total de amenidades del anuncio", num("p-namen", int(med["n_amenities"]), 0, 120),
                  "Incluye todas, no solo las de la lista."),
            campo("Noches mínimas", num("p-minnoches", int(med["minimum_nights"]), 1, 365)),
            campo("Noches máximas", num("p-maxnoches", int(med["maximum_nights"]), 1, 1125)),
            campo("Noches libres próximos 30 días", num("p-disp30", int(med["availability_30"]), 0, 30)),
            campo("Días disponibles al año", num("p-disp365", int(med["availability_365"]), 0, 365)),
            campo("Años como anfitrión", num("p-anios", 0, 0, 20, 0.5)),
            campo("Anuncios del anfitrión", num("p-nanuncios", 1, 1, 1000)),
            campo("Reseñas en otros anuncios del anfitrión", num("p-histn", 0, 0, 500),
                  "0 si es un anfitrión nuevo o no lo sabe."),
            campo("% de otros anuncios del anfitrión bien calificados", num("p-histrate", 60, 0, 100),
                  "Solo aplica si el anfitrión tiene otros anuncios con reseñas."),
            campo("Ubicación del anfitrión", dcc.Dropdown(
                id="p-hostloc", value="Hawaii", clearable=False,
                options=opciones(REF["host_location_groups"],
                                 {"Hawaii": "Vive en Hawái", "Elsewhere": "Vive fuera",
                                  "Unknown": "No informa"}))),
            campo("Perfil del anfitrión", dcc.Checklist(
                id="p-host", className="check",
                options=[{"label": " Tiene foto de perfil", "value": "pic"},
                         {"label": " Identidad verificada", "value": "ver"},
                         {"label": " Tiene licencia", "value": "lic"}],
                value=["pic", "ver", "lic"])),
        ], className="form-rejilla"),
    ], className="avanzadas"),
    html.Button("Evaluar anuncio", id="btn-evaluar", n_clicks=0, className="boton"),
], className="tarjeta")

tab_evaluar = html.Div([
    html.P("¿Encontró un anuncio, pero tiene pocas reseñas? Elija un modelo, ingrese los datos "
           "del anuncio y presione «Evaluar anuncio» para saber si probablemente será una "
           "buena estadía y si su precio es razonable.", className="intro"),
    html.Div([
        campo("Modelo", dcc.Dropdown(
            id="sel-modelo", clearable=False, value=REGISTRO[0].id,
            options=[{"label": m.nombre + ("" if m.disponible else "  (pendiente)"),
                      "value": m.id, "disabled": not m.disponible} for m in REGISTRO])),
        html.Div(id="desc-modelo", className="campo-ayuda"),
    ], className="tarjeta selector"),
    html.Div([
        formulario,
        html.Div([
            html.H3("Resultado"),
            dcc.Loading(html.Div(id="resultado",
                                 children=html.Div("Presione «Evaluar anuncio».",
                                                   className="vacio"))),
        ], className="tarjeta"),
    ], className="rejilla-evaluar"),
])

tab_ayuda = html.Div([html.Div([
    html.H3("Cómo usar el tablero"),
    html.Ol([
        html.Li([html.B("Buscar alojamiento: "),
                 "filtre por isla, tipo de alojamiento, presupuesto y huéspedes. El mapa "
                 "muestra cada anuncio (coloreado por probabilidad de ser recomendado o por "
                 "precio), las cajas comparan el precio de cada zona, la gráfica de amenidades "
                 "muestra cuáles son más comunes en los alojamientos recomendados y la tabla "
                 "lista los mejores anuncios para usted."]),
        html.Li([html.B("Evaluar un anuncio: "),
                 "escoja un modelo, ingrese los datos del anuncio que le interesa y presione "
                 "«Evaluar anuncio». Verá la probabilidad de que sea recomendado, cómo se "
                 "compara su precio con anuncios similares y qué le falta frente a los "
                 "alojamientos recomendados."]),
    ]),
    html.H3("¿Qué significa «recomendado»?"),
    html.P("Un anuncio es recomendado si tiene al menos 5 reseñas, al menos una en los "
           "últimos 12 meses, calificación general ≥ 4.8 y calificación de valor ≥ 4.7. "
           "El modelo estima esa probabilidad a partir de las características del anuncio; "
           "se considera recomendado cuando supera el umbral del modelo (0.61 para Uniandes)."),
    html.H3("Modelos"),
    html.Ul([html.Li([html.B(m.nombre + ": "), m.descripcion,
                      "" if m.disponible else " (aún no disponible)"]) for m in REGISTRO]),
    html.H3("Datos"),
    html.P(f"Inside Airbnb, Hawái · {len(DF):,} anuncios después de la limpieza.".replace(",", ".")),
], className="tarjeta")])

def _ayuda_modelos():
    bloques = []
    for m in REGISTRO:
        estado = "" if m.disponible else " (aún no disponible)"
        if m.tipo == "clasificacion":
            lectura = ("Devuelve la probabilidad (0 a 100 %) de que el anuncio sea «recomendado». "
                       "Si supera el umbral del modelo"
                       + (f" ({m.umbral:.2f})" if m.umbral else "")
                       + " se muestra «Recomendado». Todos los datos del formulario "
                         "(incluido el precio del anuncio) alimentan al modelo.")
        else:
            lectura = ("Devuelve el precio justo estimado por noche (USD) para un alojamiento con "
                       "esas características y lo compara con el precio del anuncio. Aquí el campo "
                       "«Precio por noche» NO entra al modelo: solo se usa para decir si el "
                       "anuncio está caro o barato frente a la estimación.")
        bloques.append(html.Div([
            html.H4(m.nombre + estado),
            html.P(m.descripcion),
            html.P([html.B("Cómo leer el resultado: "), lectura]),
        ], className="ayuda-modelo"))
    return bloques


modal_ayuda = html.Div([
    html.Div(className="modal-fondo", id="modal-fondo", n_clicks=0),
    html.Div([
        html.Div([
            html.H2("¿Cómo funciona el tablero?"),
            html.Button("✕", id="btn-cerrar-ayuda", n_clicks=0, className="boton-cerrar",
                        title="Cerrar"),
        ], className="modal-cabecera"),
        html.Div([
            html.H3("1. Pestaña «Buscar alojamiento»"),
            html.P("Sirve para encontrar dónde quedarse. Estos filtros afectan a TODO lo que ve "
                   "en la pestaña (indicadores, mapa, gráficas y tabla):"),
            html.Ul([
                html.Li([html.B("Isla y Tipo de alojamiento: "),
                         "puede escoger varias opciones; vacío significa «todas»."]),
                html.Li([html.B("Presupuesto por noche: "),
                         "arrastre los dos extremos para fijar el mínimo y el máximo en USD."]),
                html.Li([html.B("Huéspedes (mínimo): "),
                         "muestra solo anuncios con capacidad para al menos ese número de personas."]),
                html.Li([html.B("Mapa: "),
                         "cada punto es un anuncio. Cambie el color entre «probabilidad de ser "
                         "recomendado» (azul = más probable) y «precio por noche»."]),
                html.Li([html.B("Amenidades: "),
                         "elija entre diferencia en % de recomendados o en precio mediano."]),
                html.Li([html.B("Tabla de sugeridos: "),
                         "los 25 anuncios con mayor probabilidad de ser recomendados dentro de sus "
                         "filtros (máximo 5 por zona). Puede ordenarla con las flechas de cada "
                         "columna y abrir el anuncio en Airbnb."]),
            ]),
            html.H3("2. Pestaña «Evaluar un anuncio»"),
            html.Ol([
                html.Li("Elija el modelo en la lista desplegable."),
                html.Li("Llene los datos del anuncio que está mirando en Airbnb (isla, barrio, "
                        "tipo, precio, capacidad, baños y amenidades). Los datos del anfitrión y "
                        "de disponibilidad son opcionales; si no los sabe, deje los valores por "
                        "defecto."),
                html.Li("Presione «Evaluar anuncio». El resultado cambia según el tipo de modelo "
                        "(ver abajo) y siempre incluye la comparación de precio con anuncios "
                        "similares de la misma zona."),
            ]),
            html.P("Los filtros de la primera pestaña no se aplican al formulario: cada pestaña "
                   "funciona por separado.", className="campo-ayuda"),
            html.H3("3. Modelos y cómo leerlos"),
            html.P([html.B("Clasificación (recomendado / no recomendado): "),
                    "un anuncio es «recomendado» si tiene al menos 5 reseñas, una en los últimos "
                    "12 meses, calificación ≥ 4.8 y valor ≥ 4.7. Sirve sobre todo para anuncios "
                    "nuevos, que todavía no tienen suficientes reseñas."]),
            html.P([html.B("Regresión (precio): "),
                    "estima cuánto debería costar una noche según las características. Sirve "
                    "para saber si un anuncio está caro o barato."]),
            *_ayuda_modelos(),
        ], className="modal-cuerpo"),
    ], className="modal-caja"),
], id="modal-ayuda", className="modal oculto")


app.layout = html.Div([
    html.Header([
        html.Div([html.H1("Airbnb Hawái"),
                  html.Div("Encuentre alojamientos recomendados según su presupuesto",
                           className="subtitulo")]),
        html.Div([
            html.Div("Reto Global Classroom · Uniandes + Tec de Monterrey", className="equipo"),
            html.Button("¿Cómo funciona?", id="btn-ayuda", n_clicks=0, className="boton-ayuda"),
        ], className="cabecera-derecha"),
    ], className="cabecera"),
    modal_ayuda,
    dcc.Tabs(id="tabs", value="explorar", className="tabs", children=[
        dcc.Tab(label="Buscar alojamiento", value="explorar", children=tab_explorar,
                className="tab", selected_className="tab-activa"),
        dcc.Tab(label="Evaluar un anuncio", value="evaluar", children=tab_evaluar,
                className="tab", selected_className="tab-activa"),
        dcc.Tab(label="Instrucciones", value="ayuda", children=tab_ayuda,
                className="tab", selected_className="tab-activa"),
    ]),
], className="contenedor")


# ------------------------------------------------------------ callback: ayuda
@app.callback(Output("modal-ayuda", "className"),
              Input("btn-ayuda", "n_clicks"), Input("btn-cerrar-ayuda", "n_clicks"),
              Input("modal-fondo", "n_clicks"), prevent_initial_call=True)
def abrir_cerrar_ayuda(*_):
    return "modal" if ctx.triggered_id == "btn-ayuda" else "modal oculto"


# ------------------------------------------------------------ callbacks: explorar
@app.callback(Output("kpis", "children"), *FILTROS)
def actualizar_kpis(islas, tipos, presupuesto, huespedes):
    d = filtrar(islas, tipos, presupuesto, huespedes)
    if d.empty:
        return [kpi("Anuncios", "0", "Amplíe los filtros")]
    ev = d["recommended"].dropna()
    nuevos = d[d["recommended"].isna()]
    return [
        kpi("Anuncios que cumplen", f"{len(d):,}".replace(",", ".")),
        kpi("Precio mediano / noche", f"US$ {d['price'].median():,.0f}"),
        kpi("Recomendados (con reseñas)", f"{ev.mean():.0%}" if len(ev) else "–",
            f"de {len(ev):,} anuncios con reseñas suficientes".replace(",", ".")),
        kpi("Nuevos que el modelo recomienda",
            f"{(nuevos['prob_uniandes'] >= UMBRAL_MAPA).sum():,}".replace(",", "."),
            f"de {len(nuevos):,} con pocas reseñas".replace(",", ".")),
    ]


def figura_vacia(texto="No hay anuncios con estos filtros."):
    fig = go.Figure()
    fig.update_layout(**PLANTILLA, xaxis_visible=False, yaxis_visible=False,
                      annotations=[dict(text=texto, showarrow=False, font_size=14)])
    return fig


@app.callback(Output("g-mapa", "figure"), *FILTROS, Input("mapa-color", "value"))
def mapa(islas, tipos, presupuesto, huespedes, color):
    d = filtrar(islas, tipos, presupuesto, huespedes)
    if d.empty:
        return figura_vacia()
    if len(d) > 8000:
        d = d.sample(8000, random_state=1)
    d = d.assign(tipo=d["room_type"].map(TRAD_ROOM))
    # Zoom según la extensión de los anuncios filtrados (todas las islas ≈ 6)
    ext = max(d["latitude"].max() - d["latitude"].min(),
              d["longitude"].max() - d["longitude"].min(), 0.05)
    zoom = float(np.clip(np.log2(360 / ext) + 0.6, 5.5, 12))
    if color == "price":
        escala, rango, titulo = "Viridis", (d["price"].quantile(.05), d["price"].quantile(.95)), "USD"
    else:
        escala, rango, titulo = "RdYlBu", (0, 1), "Prob."
    fig = px.scatter_map(
        d, lat="latitude", lon="longitude", color=color, range_color=rango,
        color_continuous_scale=escala, opacity=.7, zoom=zoom,
        center={"lat": float(d["latitude"].mean()), "lon": float(d["longitude"].mean())},
        hover_name="neighbourhood",
        hover_data={"price": ":$,.0f", "tipo": True, "prob_uniandes": ":.2f",
                    "latitude": False, "longitude": False},
        labels={"price": "Precio", "prob_uniandes": "Prob. recomendado", "tipo": "Tipo"},
        map_style=os.getenv("MAP_STYLE", "carto-positron"),
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(**PLANTILLA, coloraxis_colorbar=dict(title=titulo))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


@app.callback(Output("g-precio", "figure"), *FILTROS)
def precio_barrio(islas, tipos, presupuesto, huespedes):
    d = filtrar(islas, tipos, presupuesto, huespedes)
    conteo = d["neighbourhood"].value_counts()
    d = d[d["neighbourhood"].isin(conteo[conteo >= 20].index)]
    if d.empty:
        return figura_vacia("Muy pocos anuncios por zona con estos filtros.")
    orden = d.groupby("neighbourhood")["price"].median().sort_values(ascending=False).index[:15]
    d = d[d["neighbourhood"].isin(orden)]
    fig = px.box(d, x="price", y="neighbourhood", category_orders={"neighbourhood": list(orden)},
                 points=False, color_discrete_sequence=[AZUL],
                 labels={"price": "Precio por noche (USD)", "neighbourhood": ""})
    fig.update_layout(**PLANTILLA)
    fig.update_xaxes(range=[0, d["price"].quantile(.97) if len(d) else 1000])
    return fig


@app.callback(Output("g-amen", "figure"), *FILTROS, Input("amen-metrica", "value"))
def amenidades(islas, tipos, presupuesto, huespedes, metrica):
    d = filtrar(islas, tipos, presupuesto, huespedes)
    filas = []
    for col, nombre in AMENIDADES.items():
        con, sin = d[d[col] == 1], d[d[col] == 0]
        if len(con) < 30 or len(sin) < 30:
            continue
        if metrica == "rec":
            v = (con["recommended"].mean() - sin["recommended"].mean()) * 100
        else:
            v = (con["price"].median() / sin["price"].median() - 1) * 100
        filas.append((nombre, v))
    if not filas:
        return figura_vacia("Muy pocos anuncios para comparar amenidades.")
    t = pd.DataFrame(filas, columns=["amenidad", "dif"]).sort_values("dif")
    eje = ("Diferencia en puntos porcentuales de recomendados" if metrica == "rec"
           else "Diferencia en el precio mediano (%)")
    fig = go.Figure(go.Bar(
        x=t["dif"], y=t["amenidad"], orientation="h",
        marker_color=np.where(t["dif"] >= 0, AZUL, CORAL),
        text=[f"{v:+.1f}" for v in t["dif"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f}<extra></extra>"))
    fig.update_layout(**PLANTILLA, xaxis_title=eje)
    return fig


@app.callback(Output("t-sugeridos", "data"), Output("t-sugeridos", "columns"), *FILTROS)
def sugeridos(islas, tipos, presupuesto, huespedes):
    # Orden: probabilidad del modelo y luego calificación; máximo 5 por zona
    # para que la lista muestre opciones en distintos lugares.
    d = (filtrar(islas, tipos, presupuesto, huespedes)
         .assign(_p=lambda x: x["prob_uniandes"].round(2))
         .sort_values(["_p", "review_scores_rating", "number_of_reviews"], ascending=False)
         .groupby("neighbourhood").head(5).head(25))
    t = pd.DataFrame({
        "Zona": d["neighbourhood"] + " · " + d["island"].map(TRAD_ISLA),
        "Tipo": d["room_type"].map(TRAD_ROOM),
        "Huéspedes": d["accommodates"],
        "Habitaciones": d["bedrooms"],
        "Precio / noche": d["price"].round(0),
        "Prob. recomendado": (d["prob_uniandes"] * 100).round(0),
        "Reseñas": d["number_of_reviews"],
        "Calificación": d["review_scores_rating"].round(2),
        "Origen": np.where(d["recommended"].isna(), "Estimado por el modelo",
                           "Verificado por reseñas"),
        "Ver en Airbnb": "[Abrir](https://www.airbnb.com/rooms/" + d["id"].astype(str) + ")",
    })
    cols = [{"name": c, "id": c} for c in t.columns]
    for c in cols:
        if c["id"] == "Ver en Airbnb":
            c["presentation"] = "markdown"
    return t.to_dict("records"), cols


# ------------------------------------------------------------ callbacks: evaluar
@app.callback(Output("p-barrio", "options"), Output("p-barrio", "value"), Input("p-isla", "value"))
def barrios_de_isla(isla):
    bs = [b for b in REF["barrios"] if REF["isla_de_barrio"][b] == isla]
    bs = sorted(bs, key=lambda b: -(DF["neighbourhood"] == b).sum())
    return opciones(bs), bs[0]


@app.callback(Output("desc-modelo", "children"), Input("sel-modelo", "value"))
def describir_modelo(mid):
    m = POR_ID[mid]
    if m.tipo == "regresion":
        extra = (" Este modelo estima el precio justo por noche: el campo «Precio por noche» "
                 "solo se usa para compararlo con la estimación.")
    else:
        extra = (" Este modelo estima la probabilidad de ser recomendado usando todos los "
                 "datos del formulario, incluido el precio.")
    return m.descripcion + extra


def _barra_prob(p, umbral):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 40}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": AZUL if p >= umbral else CORAL},
               "threshold": {"line": {"color": "#1d2433", "width": 3}, "value": umbral * 100},
               "steps": [{"range": [0, umbral * 100], "color": "#f3f4f6"},
                         {"range": [umbral * 100, 100], "color": "#e6f1f5"}]}))
    fig.update_layout(**PLANTILLA, height=250)
    fig.update_layout(margin=dict(l=30, r=30, t=20, b=20))
    return fig


@app.callback(
    Output("resultado", "children"),
    Input("btn-evaluar", "n_clicks"),
    State("sel-modelo", "value"), State("p-barrio", "value"), State("p-room", "value"),
    State("p-prop", "value"), State("p-precio", "value"), State("p-huespedes", "value"),
    State("p-habitaciones", "value"), State("p-camas", "value"), State("p-banos", "value"),
    State("p-bano-comp", "value"), State("p-amen", "value"), State("p-namen", "value"),
    State("p-minnoches", "value"), State("p-maxnoches", "value"), State("p-disp30", "value"),
    State("p-disp365", "value"), State("p-anios", "value"), State("p-nanuncios", "value"),
    State("p-histn", "value"), State("p-histrate", "value"), State("p-hostloc", "value"),
    State("p-host", "value"),
    prevent_initial_call=True,
)
def evaluar(_, mid, barrio, room, prop, precio, huesp, hab, camas, banos, bano_comp, amen,
            namen, minn, maxn, d30, d365, anios, nanun, histn, histrate, hostloc, host):
    valores = [precio, huesp, hab, camas, banos, namen, minn, maxn, d30, d365, anios, nanun,
               histn, histrate]
    if any(v is None for v in valores) or not barrio:
        return html.Div("Revise el formulario: hay campos vacíos o fuera de rango.", className="alerta")
    host = host or []
    p = dict(neighbourhood=barrio, room_type=room, property_group=prop, price=precio,
             accommodates=huesp, bedrooms=hab, beds=camas, bathrooms=banos,
             bath_shared=bool(bano_comp), amenities=amen or [], n_amenities=namen,
             minimum_nights=minn, maximum_nights=maxn, availability_30=d30,
             availability_365=d365, host_years=anios, host_listings=nanun,
             host_hist_n=histn, host_hist_rate=histrate / 100, host_location_group=hostloc,
             host_has_profile_pic="pic" in host, host_identity_verified="ver" in host,
             has_license="lic" in host)
    m = POR_ID[mid]
    X = pd.DataFrame([construir_caracteristicas(p, REF)])
    y = float(m.predecir(X)[0])

    # Anuncios similares: mismo barrio, tipo y capacidad parecida
    sim = DF[(DF["neighbourhood"] == barrio) & (DF["room_type"] == room)
             & (DF["accommodates"].between(huesp - 1, huesp + 1))]
    if len(sim) < 10:
        sim = DF[(DF["island"] == REF["isla_de_barrio"][barrio]) & (DF["room_type"] == room)
                 & (DF["accommodates"].between(huesp - 1, huesp + 1))]
    if len(sim):
        pct = (sim["price"] < precio).mean()
        juicio = ("económico" if pct < .25 else "caro" if pct > .75 else "en el rango normal")
        texto = (f"Comparado con {len(sim)} anuncios similares (misma zona, tipo y capacidad), "
                 f"US$ {precio:,.0f} es {juicio}: es más caro que el {pct:.0%} de ellos. "
                 f"La mediana es US$ {sim['price'].median():,.0f} y la mitad central está entre "
                 f"US$ {sim['price'].quantile(.25):,.0f} y US$ {sim['price'].quantile(.75):,.0f}.")
    else:
        texto = "No hay suficientes anuncios similares para comparar el precio."
    ref_precio = html.Div([html.H4("¿Es buen precio?"), html.P(texto)], className="bloque")

    if m.tipo == "regresion":
        return html.Div([
            html.Div("Precio justo estimado por noche", className="kpi-titulo"),
            html.Div(f"US$ {y:,.0f}", className="resultado-grande"),
            html.P(f"El anuncio cobra US$ {precio:,.0f}, "
                   f"{abs(precio / y - 1):.0%} {'más' if precio > y else 'menos'} que lo estimado."),
            ref_precio,
        ])

    umbral = m.umbral or 0.5
    veredicto = "Recomendado" if y >= umbral else "No recomendado"
    lectura = ("Es probable que este alojamiento tenga muy buenas reseñas y buena relación "
               "calidad-precio." if y >= umbral else
               "Hay riesgo de que la estadía no cumpla las expectativas; compare con los "
               "sugeridos de la primera pestaña.")

    # Qué tienen los anuncios recomendados de la zona que este anuncio no tiene
    isla = REF["isla_de_barrio"][barrio]
    recs = DF[(DF["island"] == isla) & (DF["recommended"] == 1) & (DF["room_type"] == room)]
    faltan = [(nombre, recs[col].mean() * 100) for col, nombre in AMENIDADES.items()
              if col not in (amen or []) and len(recs)]
    faltan = sorted(faltan, key=lambda x: -x[1])[:6]
    if faltan:
        consejos = html.Div([
            html.H4("Lo que le falta frente a los recomendados de la isla"),
            html.Ul([html.Li(f"{n}: la tiene el {v:.0f}% de los alojamientos recomendados")
                     for n, v in faltan]),
        ], className="bloque")
    else:
        consejos = html.Div()

    return html.Div([
        html.Div([
            html.Div(veredicto, className="veredicto " + ("ok" if y >= umbral else "no")),
            html.P(lectura),
            html.Div(f"Probabilidad de ser recomendado según {m.nombre}. "
                     f"La línea negra marca el umbral del modelo ({umbral:.2f}).",
                     className="campo-ayuda"),
        ]),
        dcc.Graph(figure=_barra_prob(y, umbral), config={"displayModeBar": False}),
        ref_precio,
        consejos,
    ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8050)), debug=False)
