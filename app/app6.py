# -*- coding: utf-8 -*-
import datetime
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import visdcc

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

app = dash.Dash(__name__,  external_scripts=external_scripts, external_stylesheets=external_stylesheets)

app.layout = html.Div([
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
                                            dbc.Button("Habilitar cropper", id='cropper-enable' ,color="secondary", className="mr-1", block=True),
                                            visdcc.Run_js(id='enablecropjs'),
                                            ],
                                            width=4,
                                        ),
                                    ]),
                                ]),
                                dbc.CardBody(
                                    [
                                        html.Div(id='image-to-crop'),
                                        html.Div(id='ancho'),
                                        html.Div(id='alto'),
                                        html.Div(id='iniciox'),
                                        html.Div(id='inicioy'),
                                    ]
                                ),
                                dbc.CardFooter([
                                    html.P('Tipo de área a especificar'),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button("Mordida", id='btn-mordida', color="primary", className="mr-1", block=True),
                                            visdcc.Run_js(id='mordidajs'),
                                        ]),
                                        dbc.Col([
                                            dbc.Button("Cuña de medición", id='btn-cuna', color="primary", className="mr-1", block=True),
                                            visdcc.Run_js(id='cunajs'),
                                        ]),
                                        dbc.Col([
                                            dbc.Button("Guía", color="primary", id='btn-guia', className="mr-1", block=True),
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
                html.H1("This is the content in tab 3"),
                html.Div(id='mordida-seleccion-multiareas'),
                dcc.Graph(
                    id='example-graph',
                    figure={
                        'data': [
                            {'x': [1, 2, 3], 'y': [4, 1, 2],
                             'type': 'bar', 'name': 'SF'},
                            {'x': [1, 2, 3], 'y': [2, 4, 5],
                             'type': 'bar', 'name': u'Montréal'},
                        ],
                        'layout': {
                            'title': 'Dash Data Visualization'
                        }
                    }
                )
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
    Output('mordida-seleccion-multiareas', 'children'),
    [Input('btn-mordida', 'n_clicks')])
def myfun(x):
    if x:
        return "console.log('Aca deberia mandar a la tercera página');"
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
