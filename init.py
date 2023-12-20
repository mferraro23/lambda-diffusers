from flask import Flask, request, send_file, Response
from werkzeug.utils import secure_filename
import os
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import io

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate_image():
    image_file = request.files.get('image')
    if image_file:
        filename = secure_filename(image_file.filename)
        # Save to a directory where your script has write permissions
        image_file.save(os.path.join('imgs/', filename))

    # Call your image generation function here with the received text and image
    image_path = os.path.join('imgs/', filename)
    
    im = Image.open(image_path)
    width, height = im.size
    if width > 1024 and height > 1024:
        width = 1024
        height = 1024
    elif width > 1024 and height != width and height < width:
        width = 1024
        height = 512

    device = "cuda"

    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
    )
    sd_pipe.safety_checker = None
    sd_pipe = sd_pipe.to(device)

    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(im).to(device).unsqueeze(0)

    out = sd_pipe(inp, width=width, height=height, guidance_scale=3)

    # Save the image to a BytesIO object
    img_io = io.BytesIO()
    out["images"][0].save(img_io, 'JPEG')
    img_io.seek(0)

    # Return the image data in a flask.Response object
    return Response(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
