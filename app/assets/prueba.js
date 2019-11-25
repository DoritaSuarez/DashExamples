// import Cropper from 'cropperjs';

var alto = 0;
var ancho = 0;
const inicioX = 0;
const inicioY = 0;

var input = false
var srcExt = "Samsung.png"

function readURL(input) {
    if (input.files && input.files[0]) {
        this.input = true;

        var reader = new FileReader();

        reader.onload = function (e) {
            document.querySelector('img#prueba')
                .setAttribute('src', e.target.result);

            document.querySelector('img#blah')
                .setAttribute('src', e.target.result);

            console.log("Aqui se carga la mordida");
            // setTimeout(()=>{                
                document.querySelector('img#mordida')
                    .setAttribute('src', e.target.result);
                setTimeout(() => {
                    crop = document.querySelector("#mordida")
                    var cropper = new Cropper(crop, options);
                    data = cropper.getCropBoxData()
                    var sale = this.cropper.getCroppedCanvas().toDataURL('image/png')
                    console.log(sale)
                
                    console.log(cropper)
                    console.log(data)
                }, 100)
                
            // }, 100)
            console.log("Asignado");
        };

        this.srcExt = this.src

        reader.readAsDataURL(input.files[0]);
    }
}

setTimeout(() => {
    console.log("Fuera", this.src)
}, 10000)

var options = {
    guides: true,
    cropBoxResizable: true,
    cropBoxMovable: true,
    autoCrop: true,
    autoCropArea: 0.4,
    viewMode: 1,
    // dragMode: 'none',
    zoomable: false,
    toggleDragModeOnDblclick: true,
    aspectRatio: 0,
    preview: '.img-preview',
    cropping: true,
    ready: function (e) {
        console.log(e.type);
        this.cropper.crop();
    },
    cropstart: function (e) {
        // console.log(e.type, e.detail.action);
    },
    cropmove: function (e) {
        // console.log(e.type, e.detail.action);
    },
    cropend: function (e) {
        // console.log(e.type, e.detail.action);
        console.log("Datos del corte")
    },
    crop: () => {
        const canvas = this.cropper.getCroppedCanvas();
        this.imaeDestination = canvas.toDataURL("image/png")
    },
    zoom: function (e) {
        console.log(e.type, e.detail.ratio);
    },
    crop(event) {
        this.inicioX = event.detail.x;
        // console.log(this.inicioX)
        console.log(event.detail.x)
        this.inicioY = event.detail.y;
        this.ancho = event.detail.width;
        this.alto = event.detail.height;
        document.querySelector("div#ancho").innerText = event.detail.width.toString()
        document.querySelector("div#alto").innerText = event.detail.height.toString()
        document.querySelector("div#iniciox").innerText = event.detail.x.toString()
        document.querySelector("div#inicioy").innerText = event.detail.y.toString()
        // console.log("InicioX", this.inicioX)
    }
};

// setTimeout(() => {
//     crop = document.querySelector("#mordida")
//     var cropper = new Cropper(crop, options);
//     data = cropper.getCropBoxData()
//     var sale = this.cropper.getCroppedCanvas().toDataURL('image/png')
//     console.log(sale)

//     console.log(cropper)
//     console.log(data)
// }, 100)

function renderizarMordida() {

    var ancho = document.querySelector("div#ancho").innerText
    var alto = document.querySelector("div#alto").innerText
    var xInit = document.querySelector("div#iniciox").innerText
    var yInit = document.querySelector("div#inicioy").innerText
    var canvas = document.getElementById('canvasMordida');
    canvas.width = ancho
    canvas.height = alto
    var context = canvas.getContext('2d');
    var image = new Image();
    image.onload = function () {
        var sourceX = xInit; var sourceY = yInit;
        var sourceWidth = ancho; var sourceHeight = alto;
        var destWidth = sourceWidth; var destHeight = sourceHeight;
        var x = canvas.width / 2 - destWidth / 2; var y = canvas.height / 2 - destHeight / 2;
        context.drawImage(this, sourceX, sourceY, sourceWidth, sourceHeight, x, y, destWidth, destHeight);

        resizeTo(canvas, 200 / canvas.height)

        function resizeTo(canvas, pct) {
            var cw = canvas.width;
            var ch = canvas.height;
            var tempCanvas = document.createElement("canvas");
            var tctx = tempCanvas.getContext("2d");
            var cw = canvas.width;
            var ch = canvas.height;
            tempCanvas.width = cw;
            tempCanvas.height = ch;
            tctx.drawImage(canvas, 0, 0);
            canvas.width *= pct;
            canvas.height *= pct;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(tempCanvas, 0, 0, cw, ch, 0, 0, cw * pct, ch * pct);
        }

    };
    image.src = 'assets/Samsung.png';
}


function renderizarCuna() {

    var ancho = document.querySelector("div#ancho").innerText
    var alto = document.querySelector("div#alto").innerText
    var xInit = document.querySelector("div#iniciox").innerText
    var yInit = document.querySelector("div#inicioy").innerText
    var canvas = document.getElementById('canvasCuna');
    canvas.width = ancho
    canvas.height = alto
    var context = canvas.getContext('2d');
    var image = new Image();
    image.onload = function () {
        var sourceX = xInit; var sourceY = yInit;
        var sourceWidth = ancho; var sourceHeight = alto;
        var destWidth = sourceWidth; var destHeight = sourceHeight;
        var x = canvas.width / 2 - destWidth / 2; var y = canvas.height / 2 - destHeight / 2;
        context.drawImage(this, sourceX, sourceY, sourceWidth, sourceHeight, x, y, destWidth, destHeight);

        resizeTo(canvas, 200 / canvas.height)

        function resizeTo(canvas, pct) {
            var cw = canvas.width;
            var ch = canvas.height;
            var tempCanvas = document.createElement("canvas");
            var tctx = tempCanvas.getContext("2d");
            var cw = canvas.width;
            var ch = canvas.height;
            tempCanvas.width = cw;
            tempCanvas.height = ch;
            tctx.drawImage(canvas, 0, 0);
            canvas.width *= pct;
            canvas.height *= pct;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(tempCanvas, 0, 0, cw, ch, 0, 0, cw * pct, ch * pct);
        }

    };
    image.src = 'assets/Samsung.png';
}

function renderizarMedicion() {

    var ancho = document.querySelector("div#ancho").innerText
    var alto = document.querySelector("div#alto").innerText
    var xInit = document.querySelector("div#iniciox").innerText
    var yInit = document.querySelector("div#inicioy").innerText
    var canvas = document.getElementById('canvasMedicion');
    canvas.width = ancho
    canvas.height = alto
    var context = canvas.getContext('2d');
    var image = new Image();
    image.onload = function () {
        var sourceX = xInit; var sourceY = yInit;
        var sourceWidth = ancho; var sourceHeight = alto;
        var destWidth = sourceWidth; var destHeight = sourceHeight;
        var x = canvas.width / 2 - destWidth / 2; var y = canvas.height / 2 - destHeight / 2;
        context.drawImage(this, sourceX, sourceY, sourceWidth, sourceHeight, x, y, destWidth, destHeight);

        resizeTo(canvas, 200 / canvas.height)

        function resizeTo(canvas, pct) {
            var cw = canvas.width;
            var ch = canvas.height;
            var tempCanvas = document.createElement("canvas");
            var tctx = tempCanvas.getContext("2d");
            var cw = canvas.width;
            var ch = canvas.height;
            tempCanvas.width = cw;
            tempCanvas.height = ch;
            tctx.drawImage(canvas, 0, 0);
            canvas.width *= pct;
            canvas.height *= pct;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(tempCanvas, 0, 0, cw, ch, 0, 0, cw * pct, ch * pct);
        }

    };
    image.src = 'assets/Samsung.png';
}


function next() {
    if (this.input) {
        console.log(document.querySelector("img#mordida").innerHTML)

    } else {
        alert("Aún no has seleccionado una imagen")
    }
}

setTimeout(() => {
    console.log(document.querySelector("#obtener"))
    console.log("Hola, tansmitiendo desde DASH :) ")
}, 500)



// cambiar(){

// }


//     // create destiantion canvas
    // var canvas1 = document.createElement("canvas");
//     canvas1.width = 100;
//     canvas1.height = 100;
//     var ctx1 = canvas1.getContext("2d");
//     ctx1.rect(0, 0, 100, 100);
//     ctx1.fillStyle = 'white';
//     ctx1.fill();
//     ctx1.putImageData(imageData, 0, 0);

//     // put data to the img element
//     var dstImg = $('#newImg').get(0);
//     dstImg.src = canvas1.toDataURL("image/png");
// });



// var image = document.querySelector('#mordida');

// image.cropper({

//   guides: true,
//   cropBoxResizable: true,
//   cropBoxMovable: true,
//   autoCrop: true,
//   autoCropArea: 0.333,
//   viewMode: 1,
//   dragMode: 'none',
//   zoomable: false,
//   toggleDragModeOnDblclick: true,
//   aspectRatio: 1,
//   preview: '.img-preview',
//   crop: function(e) {
//     var html = 'X:' + Math.round(e.x) + ' Y:' + Math.round(e.y);
//     console.log("Va el html:")
//     console.log(html);
//   },
//   ready: function() {
//     var cropboxData = image.cropper('getCropBoxData'),
//         el,
//         x,
//         y;

//     image.next('.cropper-container').find('.cropper-drag-box').on('click.setCropBoxData', function(e) {
//       el = $(this);
//       x = e.pageX - el.offset().left;
//       y = e.pageY - el.offset().top;

//       image.cropper('setCropBoxData', {
//         left: x - (cropboxData.width / 2),
//         top: y - (cropboxData.height / 2)
//       });
//     });
//   }
// });
