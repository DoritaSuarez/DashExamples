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
import functools



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

    imagen = imagen.crop(global_cropped_bite)

    imo_w, imo_h = imagen.size
    img_width = imo_w
    img_height = imo_h
    scale_factor_a = 750/imo_w
    scale_factor = 200/imo_w
    # Create figure
    print(imo_w, imo_w)
    # create tuple of tuples
    x = list(map(lambda x: round(x*1/scale_factor_a), xs))
    y = list(map(lambda x: round(img_height - x*1/scale_factor_a), ys))
    coordenadas_nuevas = list(zip(x, y))
    print("Es el otro CutParams")
    print(coordenadas_nuevas)

    # Cropping image
    @vectorize([int64(float64)])
    def redondear(x):
        return x
    cut_params = coordenadas_nuevas# [(150,1100),(220,820),(450.40,800),(400,1200.18)]
    cut_params = redondear(cut_params)
    pts = np.array(cut_params)

    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect

    imagen = np.array(imagen)
    croped = imagen[y:y+h, x:x+w].copy()
    ## (2) make mask

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


def calculo(image_path, bite_params, guide_params, wedge_params, x, y):
    # print(cut_params)
    # cut_params = [(150,1100),(220,820),(450.40,800),(400,1200.18)]
    image = Image.open(image_path)
    image_bw = image.convert(mode="L")  # Transform to black and white

    imagen_bit = image_bw.crop(bite_params)

    bite = np.array(image_bw.crop(bite_params))
    guide = np.array(image_bw.crop(guide_params))
    wedge = np.array(image_bw.crop(wedge_params))

    imo_w, imo_h = imagen_bit.size
    img_width = imo_w
    img_height = imo_h
    scale_factor_a = 750/imo_w
    x = list(map(lambda x: round(x*1/scale_factor_a), x))
    y = list(map(lambda x: round(img_height - x*1/scale_factor_a), y))
    cut_params = list(zip(x, y))
    print(cut_params)

    def change_contrast(img, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))
        def contrast(c):
            return 128 + factor * (c - 128)
        return img.point(contrast)

    guide_cont = change_contrast(image, 100)
    guide_cont = np.array(guide_cont.crop(guide_params))
    guide_cont = cv2.cvtColor(guide_cont, cv2.COLOR_BGR2GRAY)


    mid = round(guide_cont.shape[0]/2)
    print(guide_cont[mid])
    print("mid")
    # print(mid)
    print(guide_cont.shape)
    zeros = [i for i, e in enumerate(guide_cont[mid]) if e == 0]
    print('ceros')
    print(np.diff(zeros))

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
    print("pix")
    print(pix)
    mm = round(np.mean(np.diff(pix)))


    wedge_w = []
    for i in range(wedge.shape[0]):
        x = list(wedge[i])
        wedge_w.append(max(x))
    pix_med_y = wedge_w.index(max(wedge_w))
    #     plt.imshow(wedge)

    pix_med_x = np.argmax(wedge[pix_med_y])

    dist = []
    prof = []
    dist_p = []
    for i in range(wedge.shape[1]):
        dist.append(np.abs(i - pix_med_x) / mm)
        dist_p.append(np.abs(i - pix_med_x))
        prof.append(wedge[pix_med_x][i])
    #     plt.plot(dist_p, prof)
    data_wedge_dist = pd.DataFrame({"d_pix": prof, "x_pix": dist_p, "x_mm": dist})

    x = np.arange(-19, 19, 0.01)

    def tomm(w):
        return -np.sqrt(361 - w * w) + 19

    #     return -1/2*np.sqrt(22201-4*w*w)
    y = list(map(tomm, x))
    y_pix = list(map(lambda w: round(w * mm), y))
    x_pix = list(map(lambda w: round(w * mm), x))
    data_wedge_depth = pd.DataFrame({"x_mill": x, "y_mill": y, "y_pix": y_pix, "x_pix": x_pix})
    wedge_dd = data_wedge_depth.merge(data_wedge_dist)
    wedge_xd = wedge_dd.groupby("x_pix").median().reset_index()
    wedge_dd = wedge_dd.groupby("d_pix").median().reset_index()

    k_pix = min(np.array(wedge_dd[wedge_dd.y_mill <= 0.05]["d_pix"]))
    w_pix = min(np.array(wedge_dd[wedge_dd.y_mill <= 0.35]["d_pix"]))

    def depth_found(dist, a):
        y = [i for i, e in enumerate(prof) if e == a]
        if (len(y) == 0):
            return "not found"
        else:
            return np.median([dist[i] for i in y])

    distances = []
    depths = []
    for i in range(255):
        distances.append(depth_found(dist, i))
        depths.append(i)
    for i in range(len(distances)):
        if (i == 0 and distances[i] == "not found"):
            aux = [x for x in distances if x != "not found"]
            distances[i] = max(aux)
        if (distances[i] == "not found"):
            distances[i] = distances[i - 1]

    @vectorize([int64(float64)])
    def redondear(x):
        return x

    cut_params = redondear(cut_params)
    pts = np.array(cut_params)
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = bite[y:y + h, x:x + w].copy()
    ## (2) make mask
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    print('dst')
    # print(dst)

    ## (4) add the white background
    # bg = np.ones_like(croped, np.uint8) * 255

    # cv2.bitwise_not(bg,bg, mask=mask)
    # plt.imshow(dst)

    # class Switch(dict):
    #     def __getitem__(self, item):
    #         for key in self.keys():  # iterate over the intervals
    #             if item in key:  # if the argument is part of that interval
    #                 return super().__getitem__(key)  # return its associated value
    #         raise KeyError(item)  # if not in any interval, raise KeyError
    #
    # switch = Switch({
    #     range(k_pix, 255): 2,
    #     range(w_pix, k_pix): 1,
    #     range(0, w_pix): 0
    # })

    def classif(x_ent):
        if x_ent < w_pix:
            return 0
        elif x_ent < k_pix:
            return 1
        else:
            return 2

    print("k_pix")

    salida = dst.copy()

    for i in range(dst.shape[0]):
        # print(dst[i])
        salida[i] = list(map(lambda x: classif(x), dst[i]))
    classification = salida

    low = 0
    contact = 0
    close = 0
    for i in range(classification.shape[0]):
        for j in range(classification.shape[1]):
            if (classification[i][j] == 0):
                low = low + 1
            if (classification[i][j] == 1):
                close = close + 1
            if (classification[i][j] == 2):
                contact = contact + 1

    area_contact = (1 / mm ** 2) * contact
    area_close = (1 / mm ** 2) * close

    clasi = classification.astype(np.uint8)
    ret, thresh = cv2.threshold(clasi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    return area_contact, area_close, n_labels


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
                            block=False),
                        html.Pre(id='selected-data1'),
                    ], width=3, style={'padding': '0px'}),
                    dbc.Col([
                        dcc.Graph(
                            id='diente2-graph',
                        ),
                        dbc.Button(
                            "Diente 2",
                            id='diente2-btn',
                            color="primary",
                            className="mr-1",
                            block=False),
                        html.Pre(id='selected-data2'),
                    ], width=3, style={'padding': '0px'}),
                    dbc.Col([
                        dcc.Graph(
                            id='diente3-graph',
                        ),
                        dbc.Button(
                            "Diente 3",
                            id='diente3-btn',
                            color="primary",
                            className="mr-1",
                            block=False),
                        html.Pre(id='selected-data3'),
                    ], width=3, style={'padding': '0px'}),
                    dbc.Col([
                        dcc.Graph(
                            id='diente4-graph',
                        ),
                        dbc.Button(
                            "Diente 4",
                            id='diente4-btn',
                            color="primary",
                            className="mr-1",
                            block=False),
                        html.Pre(id='selected-data4'),
                    ], width=3, style={'padding': '0px'}),
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
                            block=False),
                        html.Pre(id='selected-data5'),
                    ], width=3, style={'padding': '0px'}),
                    dbc.Col([
                        dcc.Graph(
                            id='diente6-graph',
                        ),
                        dbc.Button(
                            "Diente 6",
                            id='diente6-btn',
                            color="primary",
                            className="mr-1",
                            block=False),
                        html.Pre(id='selected-data6'),
                    ], width=3, style={'padding': '0px'}),
                    dbc.Col([
                        dcc.Graph(
                            id='diente7-graph',
                        ),
                        dbc.Button(
                            "Diente 7",
                            id='diente7-btn',
                            color="primary",
                            className="mr-1",
                            block=False),
                        html.Pre(id='selected-data7'),
                    ], width=3, style={'padding': '0px'}),
                    dbc.Col([
                        dcc.Graph(
                            id='diente8-graph',
                        ),
                        dbc.Button(
                            "Diente 8",
                            id='diente8-btn',
                            color="primary",
                            className="mr-1",
                            block=False),
                        html.Pre(id='selected-data8'),
                    ], width=3, style={'padding': '0px'}),
                ]),
            ])
        ],
        style={
            'fontFamily': 'system-ui',
            'box-sizing': 'border-box'
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
    [Output('selected-data1', 'children'),
     Output('diente1-graph', 'figure')],
    [Input('diente1-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')],)
def display_selected_data(n_clicks, selectedData):
    print(global_cropped_bite)
    print(global_cropped_wedge)
    print(global_cropped_guide)
    print(raw_image_path)
    x = selectedData["lassoPoints"]["x"]
    y = selectedData["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    print('paso anterior al calculo de todo')
    salida = calculo(raw_image_path, global_cropped_bite, global_cropped_guide, global_cropped_wedge ,x,y)
    print('ya está salida calculado')
    print(salida)
    torender = html.Div([
        html.H6('Contacto'),
        html.Div(round(salida[0], 4)),
        html.H6('Contacto cercano'),
        html.Div(round(salida[1], 4)),
        html.H6('Número de contactos'),
        html.Div(round(salida[2], 4)),
    ])
    # return json.dumps(selectedData, indent=2), figure
    return torender, figure


@app.callback(
    [Output('selected-data2', 'children'),
     Output('diente2-graph', 'figure')],
    [Input('diente2-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')],)
def display_selected_data(n_clicks, selectedData):
    print(global_cropped_bite)
    print(global_cropped_wedge)
    print(global_cropped_guide)
    print(raw_image_path)
    x = selectedData["lassoPoints"]["x"]
    y = selectedData["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    salida = calculo(raw_image_path, global_cropped_bite, global_cropped_guide, global_cropped_wedge ,x,y)
    torender = html.Div([
        html.H6('Contacto'),
        html.Div(round(salida[0], 4)),
        html.H6('Contacto cercano'),
        html.Div(round(salida[1], 4)),
        html.H6('Número de contactos'),
        html.Div(round(salida[2], 4)),
    ])
    return torender, figure


@app.callback(
    [Output('selected-data3', 'children'),
     Output('diente3-graph', 'figure')],
    [Input('diente3-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')],)
def display_selected_data(n_clicks, selectedData):
    print(global_cropped_bite)
    print(global_cropped_wedge)
    print(global_cropped_guide)
    print(raw_image_path)
    x = selectedData["lassoPoints"]["x"]
    y = selectedData["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    salida = calculo(raw_image_path, global_cropped_bite, global_cropped_guide, global_cropped_wedge ,x,y)
    torender = html.Div([
        html.H6('Contacto'),
        html.Div(round(salida[0], 4)),
        html.H6('Contacto cercano'),
        html.Div(round(salida[1], 4)),
        html.H6('Número de contactos'),
        html.Div(round(salida[2], 4)),
    ])
    return torender, figure


@app.callback(
    [Output('selected-data4', 'children'),
     Output('diente4-graph', 'figure')],
    [Input('diente4-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')],)
def display_selected_data(n_clicks, selectedData):
    print(global_cropped_bite)
    print(global_cropped_wedge)
    print(global_cropped_guide)
    print(raw_image_path)
    x = selectedData["lassoPoints"]["x"]
    y = selectedData["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    salida = calculo(raw_image_path, global_cropped_bite, global_cropped_guide, global_cropped_wedge ,x,y)
    torender = html.Div([
        html.H6('Contacto'),
        html.Div(round(salida[0], 4)),
        html.H6('Contacto cercano'),
        html.Div(round(salida[1], 4)),
        html.H6('Número de contactos'),
        html.Div(round(salida[2], 4)),
    ])
    return torender, figure

@app.callback(
    [Output('selected-data5', 'children'),
     Output('diente5-graph', 'figure')],
    [Input('diente5-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')],)
def display_selected_data(n_clicks, selectedData):
    print(global_cropped_bite)
    print(global_cropped_wedge)
    print(global_cropped_guide)
    print(raw_image_path)
    x = selectedData["lassoPoints"]["x"]
    y = selectedData["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    salida = calculo(raw_image_path, global_cropped_bite, global_cropped_guide, global_cropped_wedge ,x,y)
    torender = html.Div([
        html.H6('Contacto'),
        html.Div(round(salida[0], 4)),
        html.H6('Contacto cercano'),
        html.Div(round(salida[1], 4)),
        html.H6('Número de contactos'),
        html.Div(round(salida[2], 4)),
    ])
    return torender, figure


@app.callback(
    [Output('selected-data6', 'children'),
     Output('diente6-graph', 'figure')],
    [Input('diente6-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')],)
def display_selected_data(n_clicks, selectedData):
    print(global_cropped_bite)
    print(global_cropped_wedge)
    print(global_cropped_guide)
    print(raw_image_path)
    x = selectedData["lassoPoints"]["x"]
    y = selectedData["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    salida = calculo(raw_image_path, global_cropped_bite, global_cropped_guide, global_cropped_wedge ,x,y)
    torender = html.Div([
        html.H6('Contacto'),
        html.Div(round(salida[0], 4)),
        html.H6('Contacto cercano'),
        html.Div(round(salida[1], 4)),
        html.H6('Número de contactos'),
        html.Div(round(salida[2], 4)),
    ])
    return torender, figure


@app.callback(
    [Output('selected-data7', 'children'),
     Output('diente7-graph', 'figure')],
    [Input('diente7-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')],)
def display_selected_data(n_clicks, selectedData):
    print(global_cropped_bite)
    print(global_cropped_wedge)
    print(global_cropped_guide)
    print(raw_image_path)
    x = selectedData["lassoPoints"]["x"]
    y = selectedData["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    salida = calculo(raw_image_path, global_cropped_bite, global_cropped_guide, global_cropped_wedge ,x,y)
    torender = html.Div([
        html.H6('Contacto'),
        html.Div(round(salida[0], 4)),
        html.H6('Contacto cercano'),
        html.Div(round(salida[1], 4)),
        html.H6('Número de contactos'),
        html.Div(round(salida[2], 4)),
    ])
    return torender, figure


@app.callback(
    [Output('selected-data8', 'children'),
     Output('diente8-graph', 'figure')],
    [Input('diente8-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')],)
def display_selected_data(n_clicks, selectedData):
    print(global_cropped_bite)
    print(global_cropped_wedge)
    print(global_cropped_guide)
    print(raw_image_path)
    x = selectedData["lassoPoints"]["x"]
    y = selectedData["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    salida = calculo(raw_image_path, global_cropped_bite, global_cropped_guide, global_cropped_wedge ,x,y)
    torender = html.Div([
        html.H6('Contacto'),
        html.Div(round(salida[0], 4)),
        html.H6('Contacto cercano'),
        html.Div(round(salida[1], 4)),
        html.H6('Número de contactos'),
        html.Div(round(salida[2], 4)),
    ])
    return torender, figure


if __name__ == '__main__':
    app.run_server(debug=True)
