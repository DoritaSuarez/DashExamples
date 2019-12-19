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
import plotly.graph_objs as go
import json
from copy import deepcopy
from numba import vectorize, float64, int64



# external JavaScript files
external_scripts = [
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
    'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
]

app = dash.Dash(
    __name__,
    external_scripts=external_scripts,
    external_stylesheets=external_stylesheets
)

raw_image_path = ''
global_cropped_bite = None
global_cropped_wedge = None
global_cropped_guide = None

def parse_contents(contents, filename, date):
    return html.Div([
        dbc.Row([
            html.H2('Imagen a analizar'),
        ]),
        dbc.Row([
            dbc.Col([
                html.H5('Nombre del archivo'),
                html.P(filename),
                html.H5('Fecha de carga'),
                html.P(datetime.datetime.fromtimestamp(date)),
                html.H5('Contenido crudo'),
                html.P(contents[0:200] + '...', style={
                    'whiteSpace': 'pre-wrap',
                    'wordBreak': 'break-all'
                }),
            ], width=8),
            dbc.Col([
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.CardImg(id='card-img-input', src=contents, top=True),
                            ]
                        ),
                    ],
                    style={"width": "100%"},
                ),
            ], width=4),
        ])
    ])


def create_image_figure(image_path):
    # Load the image
    global raw_image_path
    raw_image_path = image_path
    imagen = Image.open(image_path)
    im_w, im_h = imagen.size
    # Create figure
    fig = go.Figure()
    # Constants
    img_width = im_w
    img_height = im_h
    scale_factor = 550/im_w
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


def create_figure_cropped_box(image_path, coords, cut_type, factor=250):
    # Load the image
    print(coords)
    imagen = Image.open(image_path)
    imo_w, imo_h = imagen.size
    print(imagen)
    # Create figure
    print(imo_w, imo_w)
    coords2 = (500, 600, 2300, 2200)
    coords3 = tuple(list(map(lambda x: (imo_w/550)*x, list(coords))))
    a, b, c, d = coords3
    coords4 = (a, imo_h-d, b, imo_h-c)
    print(coords2)
    print(coords3)
    print(coords4)
    imagen = imagen.crop(coords4)
    if cut_type == 'bite':
        global global_cropped_bite
        global_cropped_bite=coords4
    elif cut_type == 'wedge':
        global global_cropped_wedge
        global_cropped_wedge=coords4
    elif cut_type == 'guide':
        global global_cropped_guide
        global_cropped_guide = coords4
    fig = go.Figure()
    im_w, im_h = imagen.size
    print(imagen.size) 
    # Constants
    img_width = im_w
    img_height = im_h
    scale_factor = factor/im_w
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


def create_figure_cropped_lasso(image_path, xs, ys):
    # Load the image
    imagen = Image.open(image_path)
    print('imagen recien cargada')
    print(imagen)
    imagen = imagen.crop(global_cropped_bite)
    print('imagen post crop')
    print(imagen)
    imo_w, imo_h = imagen.size
    img_width = imo_w
    img_height = imo_h
    scale_factor_a = 750/imo_w
    scale_factor = 250/imo_w
    # Create figure
    print(imo_w, imo_w)
    # create tuple of tuples
    x = list(map(lambda x: round(x*1/scale_factor_a), xs))
    y = list(map(lambda x: round(img_height - x*1/scale_factor_a), ys))
    coordenadas_nuevas = list(zip(x, y))
    # Cropping image
    @vectorize([int64(float64)])
    def redondear(x):
        return x
    cut_params = coordenadas_nuevas# [(150,1100),(220,820),(450.40,800),(400,1200.18)]
    cut_params = redondear(cut_params)
    pts = np.array(cut_params)
    print("Va pts")
    print(pts)
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    print('rect')
    print(rect)
    print('imagen')
    print(imagen)
    imagen = np.array(imagen)
    croped = imagen[y:y+h, x:x+w].copy()
    ## (2) make mask
    print('croped')
    print(croped)
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    ## (3) do bit-op
    imagen = cv2.bitwise_and(croped, croped, mask=mask)
    imagen = Image.fromarray(imagen)
    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    fig = go.Figure()
    im_w, im_h = imagen.size
    print(imagen.size)
    # Constants
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

app.layout = html.Div([
    html.H1(
        'Análisis de Contactos Oclusales',
        style={
            'textAlign': 'center',
            'margin': '48px 0',
            'fontFamily': 'system-ui'
        }
    ),
    dcc.Tabs(
        id="tabs",
        children=[
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
                                        ]),
                                    ]),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id='my-graph',
                                            ),
                                        ]
                                    ),
                                    dbc.CardFooter([
                                        html.H6('Tipo de área a especificar'),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Button("Mordida", id='btn-mordida', color="primary",
                                                           className="mr-1",
                                                           block=True),
                                            ]),
                                            dbc.Col([
                                                dbc.Button("Cuña de medición", id='btn-cuna', color="primary",
                                                           className="mr-1", block=True),
                                            ]),
                                            dbc.Col([
                                                dbc.Button("Guía", color="primary", id='btn-guia', className="mr-1",
                                                           block=True),
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
                                            dcc.Graph(
                                                id='mordida-graph',
                                            ),
                                            html.Div(id='coordinates-div-mordida'),
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
                                            dcc.Graph(
                                                id='cuna-graph',
                                            ),
                                            html.Div(id='coordinates-div-cuna'),
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
                                            dcc.Graph(
                                                id='guia-graph',
                                            ),
                                            html.Div(id='coordinates-div-guia'),
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
                html.H3('Paso #3'),
                html.H2('Especificación de piezas dentales'),
                dcc.Graph(
                    id='especificacion-piezas'
                ),
                dbc.Row([]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='diente1-graph',
                        ),
                        dbc.Button(
                            "Diente 1",
                            id='diente1-btn',
                            color="primary",
                            className="mr-1",
                            block=True),
                        html.Pre(id='selected-data'),
                    ]),
                    dbc.Col([
                        dcc.Graph(
                            id='diente2-graph',
                        ),
                        dbc.Button(
                            "Diente 2",
                            id='diente2-btn',
                            color="primary",
                            className="mr-1",
                            block=True),
                    ]),
                    dbc.Col([
                        dcc.Graph(
                            id='diente3-graph',
                        ),
                        dbc.Button(
                            "Diente 3",
                            id='diente3-btn',
                            color="primary",
                            className="mr-1",
                            block=True),
                    ]),
                    dbc.Col([
                        dcc.Graph(
                            id='diente4-graph',
                        ),
                        dbc.Button(
                            "Diente 4",
                            id='diente4-btn',
                            color="primary",
                            className="mr-1",
                            block=True),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='diente5-graph',
                        ),
                        dbc.Button(
                            "Diente 5",
                            id='diente5-btn',
                            color="primary",
                            className="mr-1",
                            block=True),
                    ]),
                    dbc.Col([
                        dcc.Graph(
                            id='diente6-graph',
                        ),
                        dbc.Button(
                            "Diente 6",
                            id='diente6-btn',
                            color="primary",
                            className="mr-1",
                            block=True),
                    ]),
                    dbc.Col([
                        dcc.Graph(
                            id='diente7-graph',
                        ),
                        dbc.Button(
                            "Diente 7",
                            id='diente7-btn',
                            color="primary",
                            className="mr-1",
                            block=True),
                    ]),
                    dbc.Col([
                        dcc.Graph(
                            id='diente8-graph',
                        ),
                        dbc.Button(
                            "Diente 8",
                            id='diente8-btn',
                            color="primary",
                            className="mr-1",
                            block=True),
                    ]),
                ]),
            ])
        ],
        style={
            'fontFamily': 'system-ui'
        },
        content_style=
        {
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


@app.callback([Output('output-image-upload', 'children'),
               Output('my-graph', 'figure')],
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    # print(list_of_names[0])
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        figure = create_image_figure(list_of_names[0])
        return children, figure


@app.callback(
    [Output('coordinates-div-mordida', 'children'),
     Output('mordida-graph', 'figure'),
     Output('especificacion-piezas', 'figure')],
    [Input('btn-mordida', 'n_clicks')],
    [State('my-graph', 'selectedData')])
def display_selected_data(n_clicks, selected_data):
    data = json.dumps(selected_data, indent=2)
    print("raw_image_path")
    print(raw_image_path)
    print('type data')
    x = selected_data["range"]['x']
    y = selected_data["range"]['y']
    x.extend(y)
    x = tuple(x)
    print(x)
    figure = create_figure_cropped_box(raw_image_path, x, 'bite')
    figure_zoomed = create_figure_cropped_box(raw_image_path, x, 'bite', 750)
    return html.P(), figure, figure_zoomed


@app.callback(
    [Output('coordinates-div-cuna', 'children'),
     Output('cuna-graph', 'figure')],
    [Input('btn-cuna', 'n_clicks')],
    [State('my-graph', 'selectedData')])
def display_selected_data(n_clicks, selected_data):
    data = json.dumps(selected_data, indent=2)
    print("raw_image_path")
    print(raw_image_path)
    print('type data')
    x = selected_data["range"]['x']
    y = selected_data["range"]['y']
    x.extend(y)
    x = tuple(x)
    print(x)
    figure = create_figure_cropped_box(raw_image_path, x, 'wedge')
    return html.P(), figure


@app.callback(
    [Output('coordinates-div-guia', 'children'),
     Output('guia-graph', 'figure')],
    [Input('btn-guia', 'n_clicks')],
    [State('my-graph', 'selectedData')])
def display_selected_data(n_clicks, selected_data):
    data = json.dumps(selected_data, indent=2)
    print("raw_image_path")
    print(raw_image_path)
    print('type data')
    x = selected_data["range"]['x']
    y = selected_data["range"]['y']
    x.extend(y)
    x = tuple(x)
    print(x)
    figure = create_figure_cropped_box(raw_image_path, x, 'guide')
    return html.P(), figure


@app.callback(
    [Output('selected-data', 'children'),
     Output('diente1-graph', 'figure')],
    [Input('especificacion-piezas', 'selectedData')])
def display_selected_data(selectedData):
    print(global_cropped_bite)
    print(global_cropped_wedge)
    print(global_cropped_guide)
    print(raw_image_path)
    x = selectedData["lassoPoints"]["x"]
    y = selectedData["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    return json.dumps(selectedData, indent=2), figure


if __name__ == '__main__':
    app.run_server(debug=True)
