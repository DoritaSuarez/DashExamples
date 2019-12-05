// var scriptCrop = document.createElement('script');
// scriptCrop.setAttribute('src','https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.6/cropper.min.js');
// document.head.appendChild(scriptCrop);

var alto = 0;
var ancho = 0;
var inicioX = 0;
var inicioY = 0;
var finX = 0;
var finY = 0;

var input = false
var srcExt = "Samsung.png"


function readURL(input) {
    setTimeout(() => {
        crop = document.querySelector("#mordida")
        var cropper = new Cropper(crop, options);
        data = cropper.getCropBoxData()
        console.log(cropper)
        console.log(data)
    }, 100)
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
        this.inicioX = Math.round(event.detail.x                        )   ;
        this.inicioY = Math.round(event.detail.y                        )   ;
        this.ancho   = Math.round(event.detail.width                    )   ;
        this.alto    = Math.round(event.detail.height                   )   ;
        this.finX    = Math.round((event.detail.width + event.detail.x) )   ;
        this.finY    = Math.round((event.detail.height + event.detail.y))   ;

        document.querySelector("div#ancho").innerHTML = this.ancho.toString()
        document.querySelector("div#alto").innerHTML = this.alto.toString()
        document.querySelector("div#iniciox").innerHTML = this.inicioX.toString()
        document.querySelector("div#inicioy").innerHTML = this.inicioY.toString()
        document.querySelector("div#finx").innerHTML = this.finX.toString()
        document.querySelector("div#finy").innerHTML = this.finY.toString()
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

    image.src = document.querySelector("img#mordida").src;
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
    image.src = document.querySelector("img#mordida").src;
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
    image.src = document.querySelector("img#mordida").src;
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
