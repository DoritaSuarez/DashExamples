// var Croppr = require('croppr');
// console.log(Croppr)

// // ES6 import
// import Croppr from 'croppr';




// console.log(document.querySelector("#croppr"))

// document.getElementById('croppr').onload = function () {
//     new Cropper(this, {
//         max_width: 300, max_height: 300
//     });
// }
// var value = croppr.getValue();

// new ImageCropper("#croppr", "Samsung.jpg", {
//     max_width: 300, max_height: 300
// }

setTimeout(() => {
    crop = document.querySelector("#croppr")
    // crop.style.height = "50px"
    console.log(crop)

    var croppr = new Croppr(crop, {
        // preview: ".primero",
        // aspectRatio: 16 / 9,
        crop(event) {
            console.log(event.detail.x);
            console.log(event.detail.y);
            console.log(event.detail.width);
            console.log(event.detail.height);
            console.log(event.detail.rotate);
            console.log(event.detail.scaleX);
            console.log(event.detail.scaleY);
            
    var value = croppr.getValue();
    console.log(value)
        }
    });

    // var image = document.querySelector("ImgCanvas").cropper("getCroppedCanvas", { width: 640, height: 480 }).toDataURL('image/jpeg', 1);
    // var imagen_cortada = croppr.crop('image/png' , 1);
    // console.log("Imagen Cortdada")
    // console.log(imagen_cortada)
    // imagen = document.createElement("img")
    
    // llegada = document.querySelector("#primero")
    // llegada.
}, 100)

