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
    imagen = Image.open(image_path)
    im_w, im_h = imagen.size
    # Create figure
    fig = go.Figure()
    # Constants
    img_width = im_w
    img_height = im_h
    scale_factor = 0.25
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
                html.H2('Selección de areas'),
                dcc.Graph(
                    id='my-graph',
                    # style={'height': 300},
                    # figure=dict(
                    #     data=[
                    #         dict(
                    #             x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                    #                2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                    #             y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
                    #                350, 430, 474, 526, 488, 537, 500, 439],
                    #             name='Rest of world',
                    #             marker=dict(
                    #                 color='rgb(55, 83, 109)'
                    #             )
                    #         ),
                    #         dict(
                    #             x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                    #                2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                    #             y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,
                    #                299, 340, 403, 549, 499],
                    #             name='China',
                    #             marker=dict(
                    #                 color='rgb(26, 118, 255)'
                    #             )
                    #         )
                    #     ],
                    #     layout=dict(
                    #         title='US Export of Plastic Crap',
                    #         showlegend=True,
                    #         legend=dict(
                    #             x=0,
                    #             y=1.0
                    #         ),
                    #         margin=dict(l=40, r=0, t=40, b=30)
                    #     )
                    # )
                ),
                html.Pre(id='selected-data'),
            ]),
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
    print(list_of_names[0])
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        figure = dict(
            data=[
                dict(
                    x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                       2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                    y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
                       350, 430, 474, 526, 488, 537, 500, 439],
                    name='Rest of world',
                    marker=dict(
                        color='rgb(55, 83, 109)'
                    )
                ),
                dict(
                    x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                       2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                    y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,
                       299, 340, 403, 549, 499],
                    name='China',
                    marker=dict(
                        color='rgb(26, 118, 255)'
                    )
                )
            ],
            layout=dict(
                title='US Export of Plastic Scrap',
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1.0
                ),
                margin=dict(l=40, r=0, t=40, b=30)
            )
        )
        figure = create_image_figure(list_of_names[0])
        return children, figure


@app.callback(
    Output('selected-data', 'children'),
    [Input('my-graph', 'selectedData')])
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)
