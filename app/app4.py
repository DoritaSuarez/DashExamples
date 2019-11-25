import dash
import dash_html_components as html


class CustomDash(dash.Dash):
    def interpolate_index(self, **kwargs):
        # Inspect the arguments by printing them
        print(kwargs)
        return '''
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta http-equiv="X-UA-Compatible" content="ie=edge">
                <title>Document</title>
                <!-- Hojas de estilo -->
                <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
                    integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
                <!-- Scripts -->
                <script src="prueba.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.6/cropper.min.js"></script>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.6/cropper.min.css">
                <link rel="stylesheet" href="assets/style_marks.css">
                <link rel="stylesheet" href="assets/typography.css">
                <title>My App</title>
            </head>
            <body>
                <body>
    <div>

        <div class="barra">
            <img class="logo" src="assets/logo_CRI.png" alt="">
            <img class="logo" src="assets/logo_Baylor.png" alt="">
            <img class="logo" src="assets/logo_CES.png" alt="">
        </div>

        <div>
            <div class="container">
                <div class="row">
                    <div class="col-sm">
                        <h1 class="text-primary"> Bienvenido </h1>
                        <p class="text-secondary">
                            Lorem ipsum dolor, sit amet consectetur adipisicing elit. Provident enim ut nisi libero
                            aspernatur consectetur pariatur tenetur maiores vel quia eos corrupti facilis, laboriosam
                            itaque
                            eius unde
                            dolor magni molestiae.
                        </p>
                    </div>
                    <div class="col-sm">
                        <div class="container">
                            <img id="blah" src="assets/tooth.svg" alt="your image" />
                        </div>
                    </div>


                    <div class="col-sm">
                        <div>
                            <input type='file' onchange="readURL(this);" />
                        </div>
                    </div>
                </div>
                <div>
                    <button onclick="next()" class="btn btn-outline-primary">
                        <a id="next">Continuar</a>
                    </button>
                </div>

            </div>
        </div>

        <img src="assets/tooth.svg" alt="" id="prueba">

        <div>
            <div class="row">
                <div class="col-sm-4">
                    <div class="card card-twitter">
                        <div class="card-cont" id= "mordidadiv">
                            <img src="assets/tooth.svg" id="mordida" class="imagenes" alt="">
                        </div>
                    </div>
                    <div id="ancho"></div>
                    <div id="alto"></div>
                    <div id="iniciox"></div>
                    <div id="inicioy"></div>
                </div>

                <div class="col-sm-7">
                    <h3 class="text-primary">Registro de Mordida</h3>
                    <div class="text-secondary">
                        A continuación selecciona el area correspondiente a la mordida, la cuña de medicion y la guia.
                        <br>
                        La cuña debe ser seleccionada casi en su totalidad.
                    </div>
                    <div class="row">
                        <div class="col-sm-4">
                            <div class="card-cont">
                                <canvas id="canvasMordida"></canvas>
                            </div>
                            <div>
                                <button id="obtener" onclick="renderizarMordida()"
                                    class="btn btn-outline-primary">Registrar
                                    mordida</button>
                            </div>
                        </div>
                        <div class="col-sm-4">
                            <div class="card-cont">
                                <canvas id="canvasCuna"></canvas>
                            </div>
                            <div>
                                <button id="obtener" onclick="renderizarCuna()"
                                    class="btn btn-outline-primary">Registrar
                                    cuña</button>
                            </div>
                        </div>
                        <div class="col-sm-4">
                            <div class="card-cont">
                                <canvas id="canvasMedicion"></canvas>
                            </div>
                            <div>
                                <button id="obtener" onclick="renderizarMedicion()"
                                    class="btn btn-outline-primary">Registrar
                                    Medicion</button>
                            </div>
                        </div>
                    </div>
                    <div class="cat">
                    </div>
                </div>
            </div>
        </div>

        <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"
            integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo"
            crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"
            integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1"
            crossorigin="anonymous"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"
            integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM"
            crossorigin="anonymous"></script>

        <script>


        </script>
    </div>
                <div id="custom-header">My custom header</div>
                {app_entry}
                {config}
                {scripts}
                {renderer}
                <div id="custom-footer">My custom footer</div>
            </body>
        </html>
        '''.format(
            app_entry=kwargs['app_entry'],
            config=kwargs['config'],
            scripts=kwargs['scripts'],
            renderer=kwargs['renderer'])

app = CustomDash()

app.layout = html.Div('Simple Dash App')

if __name__ == '__main__':
    app.run_server(debug=True)