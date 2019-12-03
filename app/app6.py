# -*- coding: utf-8 -*-
import datetime
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import visdcc

from PIL import Image, ImageOps, ImageChops, ImageFilter

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

app = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)

table_header = [
    html.Thead(html.Tr([html.Th("Dato"), html.Th("Valor"), html.Th("Confirmar valor")]))
]

row1 = html.Tr([html.Td("Ancho"),
                html.Td(html.Div(id='ancho')),
                html.Td(dcc.Input(id='ancho-input', type='text', value=''))
                ])
row2 = html.Tr([html.Td("Alto"),
                html.Td(html.Div(id='alto')),
                html.Td(dcc.Input(id='alto-input', type='text', value=''))
                ])
row3 = html.Tr([html.Td("Inicio X"),
                html.Td(html.Div(id='iniciox')),
                html.Td(dcc.Input(id='iniciox-input', type='text', value=''))
                ])
row4 = html.Tr([html.Td("Inicio Y"),
                html.Td(html.Div(id='inicioy')),
                html.Td(dcc.Input(id='inicioy-input', type='text', value=''))
                ])

table_body = [html.Tbody([row1, row2, row3, row4])]

table = dbc.Table(table_header + table_body, bordered=True)

image = Image.open('E:\Miguel Orjuela\PROYECTOS\CES App\Samsung.jpg')
image_bw = image.convert(mode="L")  # Transform to black and white
# print(image_bw.size) # [0]: width in pixels [1]: height in pixels

app.layout = html.Div([
    dcc.Store(id='memory-output'),
    html.H1('Análisis de Contactos Oclusales', style={
        'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'system-ui'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Paso #1', children=[
            html.H3('Paso #1'),
            html.H2('Carga de imágen'),
            html.P('Por favor selecciona o arrastra la imágen a analizar. Recuerda que la imágen debe contener el '
                   'registro de mordida, el circulo y la regleta de calibración.'),
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
            html.P('A continuación delimita el área correspondiente a la mordida, la cuña de medicion y la guia.'),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            [
                                dbc.CardHeader([
                                    dbc.Row([
                                        dbc.Col(
                                            html.H4("Imágen cargada"),
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
                html.H2('Cálculos y configuración de áreas para análisis de profundidad.'),
                html.P('A continuación se imprimen gráficos de prueba para verificar el proceso.'),
                # TODO: Imprimir print(image_bw.size)
                html.P("Table placeholder", id='coord-x-ftw'),
                html.H4("Dimensiones de la imagen"),
                html.P(image_bw.size),
            ])
        ]),
        dcc.Tab(label='Paso #4', children=[
            html.Div([
                html.H1("This is the content in tab 3"),
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


def parse_contents(contents, filename, date):
    return html.Div([
        html.H4('Imágen a analizar'),
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


@app.callback(Output('coord-x-ftw', 'children'),
              [Input('ancho-input', 'value'),
               Input('alto-input', 'value'),
               Input('iniciox-input', 'value'),
               Input('inicioy-input', 'value'),
               ])
def updateholita(a, b, c, d):

    atable_header = [
        html.Thead(html.Tr([html.Th("Coordenada"), html.Th("Valor")]))
    ]

    arow1 = html.Tr([html.Td("ancho"), html.Td(a)])
    arow2 = html.Tr([html.Td("alto"), html.Td(b)])
    arow3 = html.Tr([html.Td("iniciox"), html.Td(c)])
    arow4 = html.Tr([html.Td("inicioy"), html.Td(d)])

    atable_body = [html.Tbody([arow1, arow2, arow3, arow4])]

    return dbc.Table(atable_header + atable_body, bordered=True)


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


# @app.callback(
#     Output('mordida-seleccion-multiareas', 'children'),
#     [Input('btn-mordida', 'n_clicks')])
# def myfun(x):
#     if x:
#         return "console.log('Este es el banner que sale cuando se le da a btn-mordida');"
#     return ""


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
