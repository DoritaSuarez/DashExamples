# -*- coding: utf-8 -*-
import datetime
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
# import visdcc
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
import urllib.parse
import scipy
from scipy import ndimage

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
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)

raw_image_path = ''
global_cropped_bite = None
global_cropped_wedge = None
global_cropped_guide = None

contact_init_table = pd.DataFrame(columns=["tooth_id", "contact_id", "contact_area", "class"])
contact_tables = [pd.DataFrame([[f"Tooth {index}", 0, 0, "undefined"]], columns=["tooth_id", "contact_id", "contact_area", "class"]) for index in range(1, 9)]

def parse_contents(contents, filename, date):
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3('Selected image'),
                html.H6('File name'),
                html.P(filename),
                html.H6('Load date'),
                html.P(datetime.datetime.fromtimestamp(date)),
                html.H6('Raw content (binary)'),
                html.P(contents[0:200] + '...', style={
                    'whiteSpace': 'pre-wrap',
                    'wordBreak': 'break-all'
                }),
            ], width=8, align="center"),
            dbc.Col([
                dbc.Card(
                    [
                        dbc.CardImg(id='card-img-input', src=contents, top=True)
                    ],
                    style={"widthb  ": "100%"},
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
    scale_factor = 550 / im_w
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
    coords3 = tuple(list(map(lambda x: (imo_w / 550) * x, list(coords))))
    a, b, c, d = coords3
    coords4 = (a, imo_h - d, b, imo_h - c)
    print(coords2)
    print(coords3)
    print(coords4)
    imagen = imagen.crop(coords4)
    if cut_type == 'bite':
        global global_cropped_bite
        global_cropped_bite = coords4
    elif cut_type == 'wedge':
        global global_cropped_wedge
        global_cropped_wedge = coords4
    elif cut_type == 'guide':
        global global_cropped_guide
        global_cropped_guide = coords4
    fig = go.Figure()
    im_w, im_h = imagen.size
    print(imagen.size)
    # Constants
    img_width = im_w
    img_height = im_h
    scale_factor = factor / im_w
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
    scale_factor_a = 750 / imo_w
    scale_factor = 170 / imo_w
    # Create figure
    print(imo_w, imo_w)
    # create tuple of tuples
    x = list(map(lambda x: round(x * 1 / scale_factor_a), xs))
    y = list(map(lambda x: round(img_height - x * 1 / scale_factor_a), ys))
    coordenadas_nuevas = list(zip(x, y))
    print("Es el otro CutParams")
    print(coordenadas_nuevas)

    # Cropping image
    @vectorize([int64(float64)])
    def redondear(x):
        return x

    cut_params = coordenadas_nuevas  # [(150,1100),(220,820),(450.40,800),(400,1200.18)]
    cut_params = redondear(cut_params)
    pts = np.array(cut_params)

    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect

    imagen = np.array(imagen)
    croped = imagen[y:y + h, x:x + w].copy()
    ## (2) make mask

    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    ## (3) do bit-op
    imagen = cv2.bitwise_and(croped, croped, mask=mask)
    imagen = Image.fromarray(imagen)
    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8) * 255
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


def calculo(image_path, bite_params, guide_params, wedge_params, x, y, tooth_index):
    # print('--------- BLOQUE calculo iniciado --------------------')
    # print(cut_params)
    # cut_params = [(150,1100),(220,820),(450.40,800),(400,1200.18)]
    def change_contrast(img, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))
        def contrast(c):
            return 128 + factor * (c - 128)
        return img.point(contrast)
    # print("Contraste")
    image = Image.open(image_path)
    image_bw = image.convert(mode="L")  # Transform to black and white
    imagen_bit = image_bw.crop(bite_params)
    # print("Creacion de los bits")
    # Searching contras automatic
    def searchContrast(image1, bite_params1):
        i = 0
        white = 0
        while(white < 254):
            bite_contrast = change_contrast(image1, i)
            bite_contrast1 = np.array(bite_contrast.crop(bite_params1))
            bite1 = np.array(bite_contrast1)
            bite = cv2.cvtColor(bite1, cv2.COLOR_BGR2GRAY)
            temp = bite[0][list(range(np.round(int(len(bite[0])/6))))]
            white = np.mean(temp)
            i = i+1
        return i-1*1.0

    contast_value = searchContrast(image, bite_params)
    # BITE
    bite = np.array(image_bw.crop(bite_params))
    bite_contrast = change_contrast(image, contast_value)
    bite_contrast = np.array(bite_contrast.crop(bite_params))
    bite = np.array(bite_contrast)
    bite = cv2.cvtColor(bite,cv2.COLOR_BGR2GRAY)
    # print("Cambio de contraste a bite")
    # GUIDE
    guide = np.array(image_bw.crop(guide_params))
    # WEDGE
    wedge = np.array(image_bw.crop(wedge_params))
    wedge_contrast = change_contrast(image, contast_value)
    wedge_contrast = np.array(wedge_contrast.crop(wedge_params))
    wedge = np.array(wedge_contrast)
    wedge = cv2.cvtColor(wedge,cv2.COLOR_BGR2GRAY)
    # print("Cambio de contraste a wedge")
    imo_w, imo_h = imagen_bit.size
    img_width = imo_w
    img_height = imo_h
    scale_factor_a = 750 / imo_w
    x = list(map(lambda x: round(x * 1 / scale_factor_a), x))
    y = list(map(lambda x: round(img_height - x * 1 / scale_factor_a), y))
    cut_params = list(zip(x, y))
    # print(cut_params)
    # print("Cut params:", cut_params)
        
    guide_cont = change_contrast(image, 100)
    guide_cont = np.array(guide_cont.crop(guide_params))
    guide_cont = cv2.cvtColor(guide_cont, cv2.COLOR_BGR2GRAY)

    mid = round(guide_cont.shape[0] / 2)
    print(guide_cont[mid])
    print("mid:" , mid)
    # print(mid)
    print(guide_cont.shape)
    zeros = [i for i, e in enumerate(guide_cont[mid]) if e == 0]
    print('ceros:', zeros)
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
    print("mm: ", mm)

    ## CALIBRATE WEDGE
    print("CALIBRATE WEDGE")
    wedge_mask = wedge.copy()
    mins = []
    maxs = []
    for i in range(wedge.shape[0]):
        wedge_mask[i] = np.where(wedge[i] == 0, 0, 1)
        if(sum(wedge_mask[i])!= 0):
            mins.append(np.min(np.where(wedge_mask[i] == 1)[0]))
            maxs.append(np.max(np.where(wedge_mask[i] == 1)[0]))

    cut1 = np.min(mins)-mm
    cut3 = np.max(maxs)+mm
    print("CALIBRATE WEDGE")

    mins_y = []
    maxs_y = []
    for i in range(wedge.shape[1]):
        if(sum(wedge_mask[:, i])!= 0):
            mins_y.append(np.min(np.where(wedge_mask[:,i] == 1)[0]))
            maxs_y.append(np.max(np.where(wedge_mask[:,i] == 1)[0]))

    cut2 = np.min(mins_y)-mm
    cut4 = np.max(maxs_y)+mm

    wedge_params1 = (int(wedge_params[0] + cut1),
                 int(wedge_params[1] + cut2),
                 int(wedge_params[2] - (wedge_params[2] - wedge_params[0] - cut3)),
                 int(wedge_params[3] - (wedge_params[3] - wedge_params[1] - cut4)))
    print("CALIBRATE WEDGE")

    wedge_contrast = change_contrast(image, 60)
    wedge_contrast = np.array(wedge_contrast.crop(wedge_params1))
    wedge = np.array(wedge_contrast)
    wedge = cv2.cvtColor(wedge,cv2.COLOR_BGR2GRAY)

    ### END WEDGE
    ### CONVERT DEPTH PIXEL TO MILLIMETERS
    wedge_w = []
    for i in range(wedge.shape[0]):
        x = list(wedge[i])
        wedge_w.append(max(x))
    pix_med_y = wedge_w.index(max(wedge_w))

    wedge_w_x = []

    for i in range(wedge.shape[1]):
        x = list(wedge[:, i])
        wedge_w_x.append(max(x))
    pix_med_x = wedge_w_x.index(max(wedge_w_x))
    pix_med_x = np.argmax(wedge[pix_med_y])

    dist = []
    prof = []
    dist_p = []
    for i in range(wedge.shape[1]):
        dist.append(np.abs(i - pix_med_x) / mm)
        dist_p.append(np.abs(i - pix_med_x))
        prof.append(wedge[pix_med_x][i])
    #     plt.plot(dist_p, prof)
    data_wedge_dist_x = pd.DataFrame({"d_pix": prof, "x_pix": dist_p, "x_mm": dist})
    print("data wedge dist")
    dist_y = []
    prof_y = []
    dist_p_y = []
    for i in range(wedge.shape[0]):
            dist_y.append((i - pix_med_y)/mm)
            dist_p_y.append((i - pix_med_y))
            prof_y.append(wedge[i][pix_med_y])

    data_wedge_dist_y = pd.DataFrame({"d_pix":prof_y, "x_pix":dist_p_y, "x_mm":dist_y})

    data_wedge_dist = data_wedge_dist_x.append(data_wedge_dist_y)
    # plt.plot(data_wedge_dist["x_mm"], data_wedge_dist["d_pix"])
    Promedios = data_wedge_dist.groupby("x_mm").mean().reset_index()
    print("Promedios: ", Promedios)

    x = np.arange(-19, 19, 0.01)
    def tomm(w):
        return -np.sqrt(361 - w * w) + 19
    
    Promedios["y_mm"] = list(map(tomm, Promedios["x_mm"]))
    Maximos = Promedios.groupby("y_mm").max().reset_index()

    x_temp = np.array(Maximos["y_mm"])
    val_pix = [0]
    val_mm = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.005]

    for i in val_mm:
        temp = np.max(Maximos["d_pix"][np.where(x_temp >= i)[0]])
        print(temp)
        val_pix.append(temp)
        
    val_pix.append(256)

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
    ## PATCH 2.0 INICIO
    def clasif(value, arr):
        for i in range(len(arr)-1):
            if value >= arr[i] and value <= arr[i+1]:
                return i
    
    dst_clasi = dst.copy()
    for i in range(dst_clasi.shape[0]):
        for j in range(dst_clasi.shape[1]):
            dst_clasi[i][j] = clasif(dst[i][j], val_pix)
    
    classification = dst_clasi

    def maskContact(matClas, pat):
        matClasRes = matClas.copy()
        for i in range(matClas.shape[0]):
            matClasRes[i] = np.where(matClas[i] == pat, 1, 0)
        return matClasRes

    areas_class = []
    for k in range(len(val_pix)-1):
        areas_class.append(sum(sum(maskContact(classification, k)))/mm**2)
    # areas_class
    val_mic = [">350", "350 - 300", "300 - 250", "250 - 200", "200 - 150", "150 - 100", "100 - 50", "50 - 5", "<5"]
    to_class_area = pd.DataFrame({'tooth_id': "Tooth # {}".format(tooth_index), "contact_id" : val_mic , "contact_area" : areas_class}).reset_index()
    to_class_area["class"] = to_class_area["index"]
    print("val_mic: " , val_mic)

    def blobs(ToClass, mm, class_cont, tooth_id):
        blobs, num = ndimage.label(ToClass)
        blobx_mm = []
        blobx_pix = []
        blobx_num = []
    #     temp2 = sum(sum(classification_close))
        if num > 0:
            for i in range(num):
                c_temp = (sum(sum(blobs == (i+1))))/(mm**2)
                p_temp = (sum(sum(blobs == (i+1))))
    #             print(c_temp)
                if c_temp > 0.05:
                    blobx_mm.append(c_temp)
                    blobx_pix.append(p_temp)
                    blobx_num.append(str(i+1))
        Total_area = sum(blobx_mm)
        Total_pix = sum(blobx_pix)
        blobx_pix.append(sum(blobx_pix))
        blobx_mm.append(sum(blobx_mm))
        blobx_num.append("Total " + class_cont)
        blobx_pix_df = pd.DataFrame({'tooth_id': tooth_id, 'contact_id':blobx_num, 'contact_pixel': blobx_pix, 'contact_area': blobx_mm, 'class': class_cont})
        return blobx_pix_df, num, Total_area

    results = pd.DataFrame()
    area_cont = 0
    area_close = 0
    n_labels = 0

    for i in range(to_class_area.shape[0]):
        classification_temp = maskContact(classification, i)
        class_temp = to_class_area["contact_id"][i]
        table_contact, quant_cont, area_cont = blobs(classification_temp, mm, class_temp, "Tooth # {}".format(tooth_index))
        if(i == 8):
            area_cont = area_cont + area_cont
            n_labels = n_labels + quant_cont
        else:
            area_close = area_close + area_cont
        results = results.append(table_contact)

    print(to_class_area)
    area_close = to_class_area["contact_area"][8]
    area_close = sum(to_class_area["contact_area"][list(range(1, 7))])
    tooth_results = to_class_area
    n_labels = table_contact.shape[0]-1
    print("table_contact: ", table_contact)
    print("tooth_results: ", tooth_results)
    return area_cont, area_close, n_labels, tooth_results

# area_contact, area_close, n_labels

    # return area_cont, quant_cont, area_close, quant_close, tooth_results


app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H1(
                    'Occlusal Contact Analysis',
                    style={
                        'textAlign': 'center',
                        'margin': '48px 0',
                        'fontFamily': 'system-ui'
                    }
                ), width=6, className="col-12 col-md-6"
            ),
            dbc.Col([
                html.Img(src="assets\logos\CIRsalud 2018.png",style={"max-height": "70px"}),
                html.Img(src="assets\logos\TEXAS A -M.png",style={"max-height": "70px"}),
                html.Img(src="assets\logos\Ces.jpg",style={"max-height": "70px"})
            ], width=6, className="col-12 col-md-6 d-flex justify-content-center"),
        ], style={
            "display": "flex",
            "flex-wrap": "wrap",
            "padding": "0px"
        }),
    ], style={"max-width": "1000px", "margin": "0px auto"}),
    dcc.Tabs(
        id="tabs",
        children=[
            dcc.Tab(label='Step 1', children=[
                html.H3('Step 1: Choose image'),
                # html.H2('Carga de imagen'),
                html.P(['You can choose the image by dragging the image to the load zone, or clicking on the load zone'
                        ' to select the file using your browser. Remember that the image must contain the bite printing, '
                        'the measuring wedge, and the calibration strip.']),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a File')
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
            dcc.Tab(label='Step 2', children=[
                html.H3('Step 2: Image calibration'),
                # html.H2('Especificación de áreas'),
                html.P([
                    'In the next step you need to specify the three areas related to the bite printing, the measuring wedge and the calibration strip. ',
                    'Please use the ',
                    html.Strong('Box Select'), ' tool to delimit each area.'
                ]),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                [
                                    dbc.CardHeader([
                                        dbc.Row([
                                            dbc.Col(
                                                html.H6("Loaded image"),
                                                width=8,
                                            ),
                                        ]),
                                    ]),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id='my-graph',
                                                figure={'layout': go.Layout(
                                                    xaxis={'showgrid': False},
                                                    yaxis={'showgrid': False}
                                                )}
                                                , style={"display": "none"}
                                            ),
                                        ]
                                    ),
                                    dbc.CardFooter([
                                        html.H6('Type of area selected'),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Button("Bite printing", id='btn-mordida', color="primary",
                                                           className="mr-1",
                                                           block=True),
                                            ]),
                                            dbc.Col([
                                                dbc.Button("Measuring wedge", id='btn-cuna', color="primary",
                                                           className="mr-1", block=True),
                                            ]),
                                            dbc.Col([
                                                dbc.Button("Calibration strip", color="primary", id='btn-guia', className="mr-1",
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
                                    dbc.CardHeader(html.H6("Bite printing")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id='mordida-graph'
                                                , style={"display": "none"}
                                            ),
                                            html.Div(id='coordinates-div-mordida'),
                                        ]
                                    ),
                                ],
                                style={"width": "100%", "margin-bottom": "10px"},
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H6("Measuring wedge")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id='cuna-graph'
                                                , style={"display": "none"}
                                            ),
                                            html.Div(id='coordinates-div-cuna'),
                                        ]
                                    ),
                                ],
                                style={"width": "100%", "margin-bottom": "10px"},
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H6("Calibration strip")),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id='guia-graph'
                                                , style={"display": "none"}
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
            dcc.Tab(label='Step 3', children=[
                dbc.Row([
                    html.H3('Step 3: Selection of dental pieces'),
                    # html.H2('Especificación de piezas dentales'),
                    html.P([
                        'Select tooth by tooth using the ',
                        html.Strong('Lasso Select'), ' tool. First select an area to be analyzed and then press each one of the buttons corresponding to the tooth selected. '
                        'You will be able to download the report of the analysis at the end of the tooth selection process.',
                    ]),
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("Bite printing"),
                        ]),
                        dbc.CardBody(
                            [
                                dcc.Graph(
                                    id='especificacion-piezas',
                                    figure={'layout': go.Layout(
                                        xaxis={'showgrid': False},
                                        yaxis={'showgrid': False}
                                    )}
                                    # , style={"display": "none"}
                                ),
                            ]
                        ),
                    ],
                    style={'width': '100%'}),
                ]),
                dbc.Row(
                    html.H3('Analysis results', style={'margin-top': '30px'}),
                ),
                dbc.Row(
                    dbc.CardDeck([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H3('Right'),
                                    html.H5("First premolar", className="card-title"),
                                    dcc.Graph(
                                        id='diente1-graph',
                                        style={'display': 'none'}
                                    ),
                                    html.Pre(id='selected-data1',className='card-text'),
                                    dbc.Button(
                                        "Process",
                                        id='diente1-btn',
                                        color="primary",
                                        className="mr-1",
                                        block=False
                                    ),
                                ]
                            ),
                            style={'height': 'fit-content'}
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H3('Right'),
                                    html.H5("Second premolar", className="card-title"),
                                    dcc.Graph(
                                        id='diente2-graph',
                                        style={'display': 'none'}
                                    ),
                                    html.Pre(id='selected-data2', className='card-text'),
                                    dbc.Button(
                                        "Process",
                                        id='diente2-btn',
                                        color="primary",
                                        className="mr-1",
                                        block=False
                                    ),
                                ]
                            ),
                            style={'height': 'fit-content'}
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H3('Right'),
                                    html.H5("First molar", className="card-title"),
                                    dcc.Graph(
                                        id='diente3-graph',
                                        style={'display': 'none'}
                                    ),
                                    html.Pre(id='selected-data3', className='card-text'),
                                    dbc.Button(
                                        "Process",
                                        id='diente3-btn',
                                        color="primary",
                                        className="mr-1",
                                        block=False
                                    ),
                                ]
                            ),
                            style={'height': 'fit-content'}
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H3('Right'),
                                        html.H5("Second molar", className="card-title"),
                                    dcc.Graph(
                                        id='diente4-graph',
                                        style={'display': 'none'}
                                    ),
                                    html.Pre(id='selected-data4', className='card-text'),
                                    dbc.Button(
                                        "Process",
                                        id='diente4-btn',
                                        color="primary",
                                        className="mr-1",
                                        block=False
                                    ),
                                ]
                            ),
                            style={'height': 'fit-content'}
                        ),
                    ])
                ),
                dbc.Row(
                    dbc.CardDeck([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H3('Left'),
                                    html.H5("First premolar", className="card-title"),
                                    dcc.Graph(
                                        id='diente5-graph',
                                        style={'display': 'none'}
                                    ),
                                    html.Pre(id='selected-data5',className='card-text'),
                                    dbc.Button(
                                        "Process",
                                        id='diente5-btn',
                                        color="primary",
                                        className="mr-1",
                                        block=False
                                    ),
                                ]
                            ),
                            style={'height': 'fit-content'}
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H3('Left'),
                                    html.H5("Second premolar", className="card-title"),
                                    dcc.Graph(
                                        id='diente6-graph',
                                        style={'display': 'none'}
                                    ),
                                    html.Pre(id='selected-data6', className='card-text'),
                                    dbc.Button(
                                        "Process",
                                        id='diente6-btn',
                                        color="primary",
                                        className="mr-1",
                                        block=False
                                    ),
                                ]
                            ),
                            style={'height': 'fit-content'}
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H3('Left'),
                                    html.H5("First molar", className="card-title"),
                                    dcc.Graph(
                                        id='diente7-graph',
                                        style={'display': 'none'}
                                    ),
                                    html.Pre(id='selected-data7', className='card-text'),
                                    dbc.Button(
                                        "Process",
                                        id='diente7-btn',
                                        color="primary",
                                        className="mr-1",
                                        block=False
                                    ),
                                ]
                            ),
                            style={'height': 'fit-content'}
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H3('Left'),
                                        html.H5("Second molar", className="card-title"),
                                    dcc.Graph(
                                        id='diente8-graph',
                                        style={'display': 'none'}
                                    ),
                                    html.Pre(id='selected-data8', className='card-text'),
                                    dbc.Button(
                                        "Process",
                                        id='diente8-btn',
                                        color="primary",
                                        className="mr-1",
                                        block=False
                                    ),
                                ]
                            ),
                            style={'height': 'fit-content'}
                        )
                    ], style={'margin-top': '30px'})
                ),
                dbc.Row([
                    # dbc.Col([
                    #     dcc.Graph(
                    #         id='diente1-graph',
                    #     ),
                    #     dbc.Button(
                    #         "Right - First premolar",
                    #         id='diente1-btn',
                    #         color="primary",
                    #         className="mr-1",
                    #         block=False),
                    #     html.Pre(id='selected-data1'),
                    # ], width=3, style={'padding': '0px'}),
                    # dbc.Col([
                    #     dcc.Graph(
                    #         id='diente2-graph',
                    #     ),
                    #     dbc.Button(
                    #         "Right - Second premolar",
                    #         id='diente2-btn',
                    #         color="primary",
                    #         className="mr-1",
                    #         block=False),
                    #     html.Pre(id='selected-data2'),
                    # ], width=3, style={'padding': '0px'}),
                    # dbc.Col([
                    #     dcc.Graph(
                    #         id='diente3-graph',
                    #     ),
                    #     dbc.Button(
                    #         "Right - First molar",
                    #         id='diente3-btn',
                    #         color="primary",
                    #         className="mr-1",
                    #         block=False),
                    #     html.Pre(id='selected-data3'),
                    # ], width=3, style={'padding': '0px'}),
                    # dbc.Col([
                    #     dcc.Graph(
                    #         id='diente4-graph',
                    #     ),
                    #     dbc.Button(
                    #         "Right - Second molar",
                    #         id='diente4-btn',
                    #         color="primary",
                    #         className="mr-1",
                    #         block=False),
                    #     html.Pre(id='selected-data4'),
                    # ], width=3, style={'padding': '0px'}),
                ]),
                dbc.Row([
                    # dbc.Col([
                    #     dcc.Graph(
                    #         id='diente5-graph',
                    #     ),
                    #     dbc.Button(
                    #         "Left -First premolar",
                    #         id='diente5-btn',
                    #         color="primary",
                    #         className="mr-1",
                    #         block=False),
                    #     html.Pre(id='selected-data5'),
                    # ], width=3, style={'padding': '0px'}),
                    # dbc.Col([
                    #     dcc.Graph(
                    #         id='diente6-graph',
                    #     ),
                    #     dbc.Button(
                    #         "Left -Second premolar",
                    #         id='diente6-btn',
                    #         color="primary",
                    #         className="mr-1",
                    #         block=False),
                    #     html.Pre(id='selected-data6'),
                    # ], width=3, style={'padding': '0px'}),
                    # dbc.Col([
                    #     dcc.Graph(
                    #         id='diente7-graph',
                    #     ),
                    #     dbc.Button(
                    #         "Left -First molar",
                    #         id='diente7-btn',
                    #         color="primary",
                    #         className="mr-1",
                    #         block=False),
                    #     html.Pre(id='selected-data7'),
                    # ], width=3, style={'padding': '0px'}),
                    # dbc.Col([
                    #     dcc.Graph(
                    #         id='diente8-graph',
                    #     ),
                    #     dbc.Button(
                    #         "Left -Second molar",
                    #         id='diente8-btn',
                    #         color="primary",
                    #         className="mr-1",
                    #         block=False),
                    #     html.Pre(id='selected-data8'),
                    # ], width=3, style={'padding': '0px'}),
                ]),
                dbc.Row([
                    html.H3('Summary', style={'margin-top': '30px'}),
                ]),
                dbc.Row(
                    dbc.Col(
                        dash_table.DataTable(
                            id='table',
                            columns=[{"name": i, "id": i} for i in contact_init_table.columns],
                            data=contact_init_table.to_dict('records'),
                            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                            style_cell={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'
                            },
                        )
                    )
                ),
                dbc.Row(
                    html.A(
                        'Download Data',
                        id='download-link',
                        download="rawdata.csv",
                        href="",
                        target="_blank"
                    )
                )
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


@app.callback(
    Output('download-link', 'href'),
    [Input('diente1-btn', 'n_clicks'),
     Input('diente2-btn', 'n_clicks'),
     Input('diente3-btn', 'n_clicks'),
     Input('diente4-btn', 'n_clicks'),
     Input('diente5-btn', 'n_clicks'),
     Input('diente6-btn', 'n_clicks'),
     Input('diente7-btn', 'n_clicks'),
     Input('diente8-btn', 'n_clicks'),
     ])
def update_download_link(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8):

    dff = contact_init_table
    for table in contact_tables:
        dff = dff.append(table)

    csv_string = dff.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string


@app.callback([Output('output-image-upload', 'children'),
               Output('my-graph', 'figure'),
               Output('my-graph', 'style'),],
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
        style = {"display": "block"}
        return children, figure, style


@app.callback(
    [Output('coordinates-div-mordida', 'children'),
     Output('mordida-graph', 'figure'),
     Output('especificacion-piezas', 'figure'),
     Output('mordida-graph', 'style'),
     ],
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
    style = {"display": "block"}
    return html.P(), figure, figure_zoomed, style


@app.callback(
    [Output('coordinates-div-cuna', 'children'),
     Output('cuna-graph', 'figure'),
     Output('cuna-graph', 'style')
     ],
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
    style = {"display": "block"}
    return html.P(), figure, style


@app.callback(
    [Output('coordinates-div-guia', 'children'),
     Output('guia-graph', 'figure'),
     Output('guia-graph', 'style')],
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
    style = {"display": "block"}
    return html.P(), figure, style


@app.callback(
    [Output('selected-data1', 'children'),
     Output('diente1-graph', 'figure'),
     Output('diente1-graph', 'style')],
    [Input('diente1-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')], )
def display_selected_data(n_clicks, selectedData):
    # print(global_cropped_bite)
    # print(global_cropped_wedge)
    # print(global_cropped_guide)
    # print(raw_image_path)
    torender, figure = transformLassoPoints(selectedData, 1)
    style = {"display": "block", 'height': '50%', 'width': '50%'}
    return torender, figure, style


@app.callback(
    [Output('selected-data2', 'children'),
     Output('diente2-graph', 'figure'),
     Output('diente2-graph', 'style')],
    [Input('diente2-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')], )
def display_selected_data(n_clicks, selectedData):
    torender, figure = transformLassoPoints(selectedData, 2)
    style = {"display": "block", 'height': '50%', 'width': '50%'}
    return torender, figure, style


@app.callback(
    [Output('selected-data3', 'children'),
     Output('diente3-graph', 'figure'),
     Output('diente3-graph', 'style')],
    [Input('diente3-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')], )
def display_selected_data(n_clicks, selectedData):
    torender, figure = transformLassoPoints(selectedData, 3)
    style = {"display": "block", 'height': '50%', 'width': '50%'}
    return torender, figure, style


@app.callback(
    [Output('selected-data4', 'children'),
     Output('diente4-graph', 'figure'),
     Output('diente4-graph', 'style')],
    [Input('diente4-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')], )
def display_selected_data(n_clicks, selectedData):
    torender, figure = transformLassoPoints(selectedData, 4)
    style = {"display": "block", 'height': '50%', 'width': '50%'}
    return torender, figure, style


@app.callback(
    [Output('selected-data5', 'children'),
     Output('diente5-graph', 'figure'),
     Output('diente5-graph', 'style')],
    [Input('diente5-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')], )
def display_selected_data(n_clicks, selectedData):
    torender, figure = transformLassoPoints(selectedData, 5)
    style = {"display": "block", 'height': '50%', 'width': '50%'}
    return torender, figure, style


@app.callback(
    [Output('selected-data6', 'children'),
     Output('diente6-graph', 'figure'),
     Output('diente6-graph', 'style')],
    [Input('diente6-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')], )
def display_selected_data(n_clicks, selectedData):
    torender, figure = transformLassoPoints(selectedData, 6)
    style = {"display": "block", 'height': '50%', 'width': '50%'}
    return torender, figure, style


@app.callback(
    [Output('selected-data7', 'children'),
     Output('diente7-graph', 'figure'),
     Output('diente7-graph', 'style')],
    [Input('diente7-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')], )
def display_selected_data(n_clicks, selectedData):
    torender, figure = transformLassoPoints(selectedData, 7)
    style = {"display": "block", 'height': '50%', 'width': '50%'}
    return torender, figure, style


@app.callback(
    [Output('selected-data8', 'children'),
     Output('diente8-graph', 'figure'),
     Output('diente8-graph', 'style')],
    [Input('diente8-btn', 'n_clicks')],
    [State('especificacion-piezas', 'selectedData')], )
def display_selected_data(n_clicks, selectedData):
    torender, figure = transformLassoPoints(selectedData, 8)
    style = {"display": "block", 'height': '50%', 'width': '50%'}
    return torender, figure, style


@app.callback(
    Output('table', 'data'),
    [Input('diente1-btn', 'n_clicks'),
     Input('diente2-btn', 'n_clicks'),
     Input('diente3-btn', 'n_clicks'),
     Input('diente4-btn', 'n_clicks'),
     Input('diente5-btn', 'n_clicks'),
     Input('diente6-btn', 'n_clicks'),
     Input('diente7-btn', 'n_clicks'),
     Input('diente8-btn', 'n_clicks')])
def update_table(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    torender_table = contact_init_table
    for table in contact_tables:
        # print("---------------------------- TABLES FOUND... -------------------")
        # print(button_id)
        # print(table.to_string)
        torender_table = torender_table.append(table)
        torender_table.loc[torender_table['tooth_id'] == 'Tooth # 1', 'tooth_id'] = 'RIGHT First premolar'
        torender_table.loc[torender_table['tooth_id'] == 'Tooth # 2', 'tooth_id'] = 'RIGHT Second premolar'
        torender_table.loc[torender_table['tooth_id'] == 'Tooth # 3', 'tooth_id'] = 'RIGHT First molar'
        torender_table.loc[torender_table['tooth_id'] == 'Tooth # 4', 'tooth_id'] = 'RIGHT Second molar'
        torender_table.loc[torender_table['tooth_id'] == 'Tooth # 5', 'tooth_id'] = 'LEFT First premolar'
        torender_table.loc[torender_table['tooth_id'] == 'Tooth # 6', 'tooth_id'] = 'LEFT Second premolar'
        torender_table.loc[torender_table['tooth_id'] == 'Tooth # 7', 'tooth_id'] = 'LEFT First molar'
        torender_table.loc[torender_table['tooth_id'] == 'Tooth # 8', 'tooth_id'] = 'LEFT Second molar'

    return torender_table.to_dict('records')


def transformLassoPoints(selected_data, index):
    x = selected_data["lassoPoints"]["x"]
    y = selected_data["lassoPoints"]["y"]
    figure = create_figure_cropped_lasso(raw_image_path, x, y)
    salida = calculo(raw_image_path, global_cropped_bite, global_cropped_guide, global_cropped_wedge, x, y, index)
    torender = html.Div([
        html.H6(f'Contact: {round(salida[0], 4)}'),
        # html.Div(round(salida[0], 4)),
        html.H6(f'Close contact: {round(salida[1], 4)}'),
        # html.Div(round(salida[1], 4)),
        html.H6(f'Number of contacts: {round(salida[2], 4)}')
        # html.Div(round(salida[2], 4)),
    ])

    contact_tables[index-1] = salida[3]
    # print("---------------------------- UPDATING TABLES WITH... -------------------")
    # print(contact_tables[index-1].to_string)
    return torender, figure


if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(host='0.0.0.0', debug=True)
