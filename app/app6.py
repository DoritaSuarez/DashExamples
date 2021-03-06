# -*- coding: utf-8 -*-
import datetime
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import visdcc
import importlib

from PIL import Image, ImageOps, ImageChops, ImageFilter
import numpy as np
import cv2
import pandas as pd

# drc = importlib.import_module("dash_reusable_components")
# utils = importlib.import_module("utils")


# external JavaScript files
external_scripts = [
    'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.6/cropper.min.js',
    {'src': 'assets/prueba.js'},
    {
        'src': 'https://code.jquery.com/jquery-3.3.1.slim.min.js',
        'integrity': 'sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo',
        'crossorigin': 'anonymous'
    },
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js',
        'integrity': 'sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1',
        'crossorigin': 'anonymous'
    },
    {
        'src': 'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js',
        'integrity': 'sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM',
        'crossorigin': 'anonymous'
    },
]

# external CSS stylesheets
external_stylesheets = [
    # 'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.6/cropper.min.css'
]

app = dash.Dash(__name__, external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)

table_header = [
    html.Thead(html.Tr([html.Th("Dato"), html.Th(
        "Valor"), html.Th("Confirmar valor")]))
]

row1 = html.Tr([html.Td("Ancho"),
                html.Td(html.Div(id='ancho')),
                html.Td(dcc.Input(id='ancho-input', type='text', value=''))
                ], hidden=True)
row2 = html.Tr([html.Td("Alto"),
                html.Td(html.Div(id='alto')),
                html.Td(dcc.Input(id='alto-input', type='text', value=''))
                ], hidden=True)
row3 = html.Tr([html.Td("Inicio X"),
                html.Td(html.Div(id='iniciox')),
                html.Td(dcc.Input(id='iniciox-input', type='text', value=''))
                ])
row4 = html.Tr([html.Td("Inicio Y"),
                html.Td(html.Div(id='inicioy')),
                html.Td(dcc.Input(id='inicioy-input', type='text', value=''))
                ])
row5 = html.Tr([html.Td("Fin X"),
                html.Td(html.Div(id='finx')),
                html.Td(dcc.Input(id='finx-input', type='text', value=''))
                ])
row6 = html.Tr([html.Td("Fin Y"),
                html.Td(html.Div(id='finy')),
                html.Td(dcc.Input(id='finy-input', type='text', value=''))
                ])

table_body = [html.Tbody([row1, row2, row3, row4, row5, row6])]

table = dbc.Table(table_header + table_body, bordered=True)

image = Image.open('app/Samsung.png')
image_bw = image.convert(mode="L")  # Transform to black and white

# print(image_bw.size) # [0]: width in pixels [1]: height in pixels

# TODO Ask for the values on the interface (@callback)
# --- Cambiar contraste en guia de medicion


def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)

    return img.point(contrast)


def tomm(w):
    return -1 / 4 * np.sqrt(5929 - 16 * w * w) + 19.25


def calculate_everything(a, b, c, d):
    if all(v is not None for v in [a, b, c, d]):
        # bite_params = (500, 600, 2300, 2200)
        bite_params = (a, b, c, d)
        guide_params = (30, 2400, 200, 2650)
        wedge_params = (1050, 2650, 1400, 2950)
        bite = np.array(image_bw.crop(bite_params))
        guide = np.array(image_bw.crop(guide_params))
        wedge = np.array(image_bw.crop(wedge_params))
        guide_cont = change_contrast(image, 100)
        guide_cont = np.array(guide_cont.crop(guide_params))
        guide_cont = cv2.cvtColor(guide_cont, cv2.COLOR_BGR2GRAY)
        mid = round(guide_cont.shape[0] / 2)
        zeros = [i for i, e in enumerate(guide_cont[mid]) if e == 0]
        sum_t = 0
        count = 0
        pix = []
        for i in range(len(zeros) - 1):
            if ((zeros[i + 1] - zeros[i]) == 1):
                sum_t = sum_t + zeros[i]
                count = count + 1
            else:
                pix.append(round(sum_t / count))
                sum_t = 0
                count = 0
        mm = round(np.mean(np.diff(pix)))

        wedge_w = []

        for i in range(wedge.shape[0]):
            x = list(wedge[i])
            wedge_w.append(max(x))
        pix_med_y = wedge_w.index(max(wedge_w))

        pix_med_x = np.argmax(wedge[pix_med_y])

        dist = []
        prof = []
        dist_p = []
        for i in range(wedge.shape[1]):
            dist.append(np.abs(i - pix_med_x) / mm)
            dist_p.append(np.abs(i - pix_med_x))
            prof.append(wedge[pix_med_x][i])

        data_wedge_dist = pd.DataFrame(
            {"d_pix": prof, "x_pix": dist_p, "x_mm": dist})

        x = np.arange(0.0, 10, 0.01)

        y = list(map(tomm, x))
        y_pix = list(map(lambda w: round(w * mm), y))
        x_pix = list(map(lambda w: round(w * mm), x))

        data_wedge_depth = pd.DataFrame(
            {"x_mill": x, "y_mill": y, "y_pix": y_pix, "x_pix": x_pix})

        wedge_dd = data_wedge_depth.merge(data_wedge_dist)
        wedge_xd = wedge_dd.groupby("x_pix").median().reset_index()
        wedge_dd = wedge_dd.groupby("d_pix").median().reset_index()

        k_pix = min(np.array(wedge_dd[wedge_dd.y_mill <= 0.05]["d_pix"]))
        w_pix = min(np.array(wedge_dd[wedge_dd.y_mill <= 0.35]["d_pix"]))

        # Metodo de corte de área para analizar todo lo que está adentro

        # bite = np.array(image.crop(bite_params))
        cut_params = [[150, 1100], [220, 820], [450, 800], [400, 1200]]
        pts = np.array(cut_params)
        # (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        croped = bite[y:y + h, x:x + w].copy()

        # (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        class Switch(dict):
            def __getitem__(self, item):
                for key in self.keys():  # iterate over the intervals
                    if item in key:  # if the argument is part of that interval
                        return super().__getitem__(key)  # return its associated value
                raise KeyError(item)  # if not in any interval, raise KeyError

        switch = Switch({
            range(k_pix, 255): 2,
            range(w_pix, k_pix): 1,
            range(0, w_pix): 0
        })

        classification = dst.copy()
        for i in range(dst.shape[0]):
            classification[i] = [switch[x] for x in dst[i]]

        low = 0
        contact = 0
        close = 0
        for i in range(classification.shape[0]):
            for j in range(classification.shape[1]):
                if (classification[i][j] == 0):
                    low = low + 1
                if (classification[i][j] == 1):
                    contact = contact + 1
                if (classification[i][j] == 2):
                    close = close + 1
        resultDiv = html.Div([
            html.P('bite shape: ' + str(bite.shape)),
            html.P('guide shape: ' + str(guide.shape)),
            html.P('wedge shape: ' + str(wedge.shape)),
            html.P('guide_cont shape: ' + str(guide_cont.shape)),
            html.P('mm: ' + str(mm)),
            html.P('pix_med_x: ' + str(pix_med_x)),
            html.P('pix_med_y: ' + str(pix_med_y)),
            html.P('k_pix: ' + str(k_pix)),
            html.P('w_pix: ' + str(w_pix)),
            html.P('low: ' + str(low)),
            html.P('contact: ' + str(contact)),
            html.P('close: ' + str(close)),
        ])
        return resultDiv
    return html.Div("Algún campo es nulo")

import plotly.graph_objs as go

def create_image_figure(image_path):
    # Image loading
    imagen = Image.open(image_path)
    im_w, im_h = imagen.size
    # Create figure
    fig = go.Figure()
    # Constants
    img_width = im_w
    img_height = im_h
    scale_factor = 0.5
    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )
    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )
    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )
    # Add image
    fig.add_layout_image(
        go.layout.Image(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=imagen)
    )
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return fig

figura_lista = create_image_figure()
# TODO Por aquí vamos


app.layout = html.Div([
    dcc.Store(id='memory-output'),
    html.H1('Análisis de Contactos Oclusales', style={
        'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'system-ui'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Paso #1', children=[
            html.H3('Paso #1'),
            html.H2('Carga de imagen'),
            html.P(['Por favor selecciona o arrastra la imagen a analizar. Recuerda que la imagen debe contener el '
                   'registro de mordida, el circulo y la regleta de calibración.']),
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '3px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-image-upload'),
            html.Div([

            ])
        ]),
        dcc.Tab(label='Paso #2', children=[

            html.H3('Paso #2'),
            html.H2('Especificación de áreas'),
            html.P(
                'A continuación delimita el área correspondiente a la mordida, la cuña de medicion y la guia.'),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            [
                                dbc.CardHeader([
                                    dbc.Row([
                                        dbc.Col(
                                            html.H4("Imagen cargada"),
                                            width=8,
                                        ),
                                        dbc.Col([
                                            dbc.Button("Habilitar cropper", id='cropper-enable', color="secondary",
                                                       className="mr-1", block=True),
                                            visdcc.Run_js(id='enablecropjs'),
                                        ],
                                            width=4,
                                        ),
                                    ]),
                                ]),
                                dbc.CardBody(
                                    [
                                        html.Div(id='image-to-crop'),
                                    ]
                                ),
                                dbc.CardFooter([
                                    html.H6('Datos del cropper'),
                                    table,
                                    html.H6('Tipo de área a especificar'),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button("Mordida", id='btn-mordida', color="primary", className="mr-1",
                                                       block=True),
                                            visdcc.Run_js(id='mordidajs'),
                                        ]),
                                        dbc.Col([
                                            dbc.Button("Cuña de medición", id='btn-cuna', color="primary",
                                                       className="mr-1", block=True),
                                            visdcc.Run_js(id='cunajs'),
                                        ]),
                                        dbc.Col([
                                            dbc.Button("Guía", color="primary", id='btn-guia', className="mr-1",
                                                       block=True),
                                            visdcc.Run_js(id='guiajs'),
                                        ]),
                                    ]),
                                ]),
                            ],
                            style={"width": "100%"},
                        )
                    ], width=8),
                    dbc.Col([
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H6("Mordida")),
                                dbc.CardBody(
                                    [
                                        html.Canvas(id='canvasMordida'),
                                    ]
                                ),
                            ],
                            style={"width": "100%"},
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H6("Cuña de medición")),
                                dbc.CardBody(
                                    [
                                        html.Canvas(id='canvasCuna'),
                                    ]
                                ),
                            ],
                            style={"width": "100%"},
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H6("Guía de medición")),
                                dbc.CardBody(
                                    [
                                        html.Canvas(id='canvasMedicion'),
                                    ]
                                ),
                            ],
                            style={"width": "100%"},
                        ),
                    ], width=4),
                ]),
            ]),
        ]),
        dcc.Tab(label='Paso #3', children=[
            html.Div([
                html.H3('Paso #3'),
                html.H2(
                    'Cálculos y configuración de áreas para análisis de profundidad.'),
                html.P(
                    'A continuación se imprimen gráficos de prueba para verificar el proceso.'),
                html.H4([
                    "Selección mordida ",
                    dbc.Button('Guardar mordida', id='guardar-mordida-btn'),
                    html.Span("", id='guardar-mordida-result')
                ]),
                html.P("Resultados de selección en MORDIDA", id='table-mordida'),
                html.H4("Selección cuña"),
                html.P("Resultados de selección en MORDIDA", id='table-cuna'),
                html.H4("Selección guia"),
                html.P("Resultados de selección en MORDIDA", id='table-guia'),
                html.H4("Dimensiones de la imagen"),
                html.P(str(image_bw.size[0]) + 'x' + str(image_bw.size[1])),
                html.Button(id='submit-button', n_clicks=0, children='Submit'),
                html.Div(id='calculate-everything-output'),
                # TODO: Por aquí vamos

            ])
        ]),
        dcc.Tab(label='Paso #4', children=[
            html.Div([
                html.H1("Resultados de la imagen"),
                dcc.Graph(
                    figure=figura_lista,
                    # style={'height': 300},
                    id='my-graph'
                )
            ])
        ]),
    ],
        style={
        'fontFamily': 'system-ui'
    },
        content_style={
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'padding': '44px'
    },
        parent_style={
        'maxWidth': '1000px',
        'margin': '0 auto'
    }
    )
])

app.config.suppress_callback_exceptions = True


def parse_contents(contents, filename, date):
    return html.Div([
        html.H4('Imagen a analizar'),
        dbc.Row([
            # dbc.Col([
            #     html.H5(datetime.datetime.fromtimestamp(date)),
            # ]),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5(filename),
                                dbc.CardImg(id='blah', src=contents, top=True),
                                # html.Div('Contenido crudo'),
                                # html.Pre(contents[0:200] + '...', style={
                                #     'whiteSpace': 'pre-wrap',
                                #     'wordBreak': 'break-all'
                                # }),
                            ]
                        ),
                    ],
                    style={"width": "100%"},
                ),
            ),
        ]),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        # html.Img(src=contents),
        # html.Hr(),
        # html.Div('Contenido crudo'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })
    ])


def parse_contents_to_cropper(contents, filename, date):
    return html.Div([
        html.Img(id='mordida', src=contents)
    ])


@app.callback(Output('guardar-mordida-result', 'children'),
              [Input('guardar-mordida-btn', 'n_clicks')],
              [State('input-inicio-x-mordida', 'value'),
               State('input-inicio-y-mordida', 'value'),
               State('input-fin-x-mordida', 'value'),
               State('input-fin-y-mordida', 'value')])
def save_bite_img(n_clicks, inicio_x, inicio_y, fin_x, fin_y):
    if all(w != '' for w in [inicio_x, inicio_y, fin_x, fin_y]):
        im = Image.open("/Samsung.png")
        print(im)
        width, height = im.size
        print('width' + width)
        print('height' + height)
        # print("IMG CARGADA")
        # borders = (inicio_x, inicio_y, width-fin_x, height-fin_y)
        # print(borders)
        # cropped = ImageOps.crop(im, borders)
        # cropped.save("E:/Miguel Orjuela/PROYECTOS/CES App/Samsung-cropped.png")
        return html.Span(inicio_x + inicio_y + fin_x + fin_y)
    return html.Span("Nodata")


@app.callback(Output('table-mordida', 'children'),
              [Input('btn-mordida', 'n_clicks')],
              [State('iniciox-input', 'value'),
               State('inicioy-input', 'value'),
               State('finx-input', 'value'),
               State('finy-input', 'value'),
               ])
def update_table_mordida(n_clicks, a, b, c, d):
    atable_header = [
        html.Thead(html.Tr([html.Th("Coordenada"), html.Th("Valor")]))
    ]

    arow1 = html.Tr([
        html.Td("Inicio X"),
        # html.Td(a),
        dcc.Input(id='input-inicio-x-mordida', value=a),
    ])
    arow2 = html.Tr([
        html.Td("Inicio Y"),
        # html.Td(b)
        dcc.Input(id='input-inicio-y-mordida', value=b),
    ])
    arow3 = html.Tr([
        html.Td("Fin X"),
        # html.Td(c),
        dcc.Input(id='input-fin-x-mordida', value=c),
    ])
    arow4 = html.Tr([
        html.Td("Fin Y"),
        # html.Td(d)
        dcc.Input(id='input-fin-y-mordida', value=d),
    ])

    atable_body = [html.Tbody([arow1, arow2, arow3, arow4])]

    # im = Image.open("Samsung.png")
    # width, height = im.size
    # print('width' + width)
    # print('height' + height)
    # print("IMG CARGADA")
    # borders = (inicio_x, inicio_y, width-fin_x, height-fin_y)
    # print(borders)
    # cropped = ImageOps.crop(im, borders)
    # cropped.save("E:/Miguel Orjuela/PROYECTOS/CES App/S



    return dbc.Table(atable_header + atable_body, bordered=True)


@app.callback(Output('table-cuna', 'children'),
              [Input('btn-cuna', 'n_clicks')],
              [State('iniciox-input', 'value'),
               State('inicioy-input', 'value'),
               State('finx-input', 'value'),
               State('finy-input', 'value'),
               ])
def update_table_cuna(n_clicks, a, b, c, d):
    atable_header = [
        html.Thead(html.Tr([html.Th("Coordenada"), html.Th("Valor")]))
    ]

    arow1 = html.Tr([html.Td("Inicio X"), html.Td(a)])
    arow2 = html.Tr([html.Td("Inicio Y"), html.Td(b)])
    arow3 = html.Tr([html.Td("Fin X"), html.Td(c)])
    arow4 = html.Tr([html.Td("Fin Y"), html.Td(d)])

    atable_body = [html.Tbody([arow1, arow2, arow3, arow4])]

    return dbc.Table(atable_header + atable_body, bordered=True)


@app.callback(Output('table-guia', 'children'),
              [Input('btn-guia', 'n_clicks')],
              [State('iniciox-input', 'value'),
               State('inicioy-input', 'value'),
               State('finx-input', 'value'),
               State('finy-input', 'value'),
               ])
def update_table_guia(n_clicks, a, b, c, d):
    atable_header = [
        html.Thead(html.Tr([html.Th("Coordenada"), html.Th("Valor")]))
    ]

    arow1 = html.Tr([html.Td("Inicio X"), html.Td(a)])
    arow2 = html.Tr([html.Td("Inicio Y"), html.Td(b)])
    arow3 = html.Tr([html.Td("Fin X"), html.Td(c)])
    arow4 = html.Tr([html.Td("Fin Y"), html.Td(d)])

    atable_body = [html.Tbody([arow1, arow2, arow3, arow4])]

    return dbc.Table(atable_header + atable_body, bordered=True)


def calculate_dogs(a, b, c, d):
    ia, ib, ic, idd = 0, 0, 0, 0
    if all(w != '' for w in [a, b, c, d]):
        ia = float(a)
        ib = float(b)
        ic = float(c)
        idd = float(d)
        if all(v is not None for v in [ia, ib, ic, idd]):
            return calculate_everything(ia, ib, ic, idd)
    return html.Div("Nodata")


@app.callback(Output('calculate-everything-output', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('iniciox-input', 'value'),
               State('inicioy-input', 'value'),
               State('finx-input', 'value'),
               State('finy-input', 'value'),
               ])
def update_coordenates(n_clicks, a, b, c, d):
    return calculate_dogs(a, b, c, d)


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('image-to-crop', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents_to_cropper(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(
    Output('enablecropjs', 'run'),
    [Input('cropper-enable', 'n_clicks')])
def myfun(x):
    if x:
        return "readURL();"
    return ""


@app.callback(
    Output('mordidajs', 'run'),
    [Input('btn-mordida', 'n_clicks')])
def myfun(x):
    if x:
        return "renderizarMordida();"
    return ""


@app.callback(
    Output('cunajs', 'run'),
    [Input('btn-cuna', 'n_clicks')])
def myfun(x):
    if x:
        return "renderizarCuna();"
    return ""


@app.callback(
    Output('guiajs', 'run'),
    [Input('btn-guia', 'n_clicks')])
def myfun(x):
    if x:
        return "renderizarMedicion();"
    return ""


if __name__ == '__main__':
    app.run_server(debug=True)
