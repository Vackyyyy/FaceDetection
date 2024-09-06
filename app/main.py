from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import torch
import cv2
from torchvision import transforms
from facenet_pytorch import MTCNN
from pathlib import Path
# MODNet imports
from app.MODNet.src.models.modnet import MODNet
import os

app = FastAPI()

# Mount static files (e.g., for serving uploaded images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Load MODNet model on CPU
def load_modnet_model(ckpt_path):
    modnet = MODNet(backbone_pretrained=False)
    modnet = torch.nn.DataParallel(modnet)
    modnet.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    modnet.eval()
    return modnet

# Preprocessing function
def preprocess_image(image):
    original_size = image.size
    image = image.resize((512, 512), Image.BILINEAR)
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image, original_size

# Postprocessing function
def postprocess_alpha(alpha, original_size):
    alpha = alpha.squeeze().cpu().numpy()
    alpha = cv2.resize(alpha, original_size, interpolation=cv2.INTER_LINEAR)
    return alpha

# Function to refine the alpha matte and smooth the edges
def refine_alpha(alpha):
    # Apply a binary threshold to the alpha to remove weak regions
    alpha_threshold = (alpha > 0.1).astype(np.uint8) * 255

    # Use GaussianBlur to smooth the edges of the alpha mask
    alpha_blurred = cv2.GaussianBlur(alpha_threshold, (7, 7), 0)

    # Normalize the alpha values back between 0 and 1 after blurring
    alpha_smoothed = alpha_blurred / 255.0

    return alpha_smoothed

# Function to crop the above neck region with face detection
def crop_above_neck(image, alpha):
    # Convert image from PIL to NumPy without changing the color space (keep it RGB)
    original_image = np.array(image)  # Image is still in RGB
    alpha = refine_alpha(alpha)
    alpha = (alpha * 255).astype(np.uint8)

    # Create a white background (RGB format)
    white_background = np.ones_like(original_image) * 255
    result = np.where(alpha[..., None] > 0, original_image, white_background)

    # Use MTCNN for face detection
    mtcnn = MTCNN(keep_all=False)
    boxes, _ = mtcnn.detect(image)

    # Ensure exactly one face is detected
    if boxes is None or len(boxes) != 1:
        raise HTTPException(status_code=400, detail="The image must contain exactly one face.")

    x1, y1, x2, y2 = boxes[0]
    padding = int((y2 - y1) * 0.3)
    padding_below = int((y2 - y1) * 0.1)
    top_region = max(int(y1 - padding), 0)
    bottom_region = min(int(y2 - padding_below), original_image.shape[0])

    face_and_hair_region = result[top_region:bottom_region, :, :]

    full_size_face_image = np.ones_like(original_image) * 255
    full_size_face_image[top_region:bottom_region, :, :] = face_and_hair_region

    # Resize the output image to 600x600 while keeping RGB format
    resized_image = cv2.resize(full_size_face_image, (600, 600), interpolation=cv2.INTER_LINEAR)

    return resized_image

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image = Image.open(file.file).convert("RGB")  # Ensure image is in RGB format
    modnet = load_modnet_model('modnet_photographic_portrait_matting.ckpt')

    im, original_size = preprocess_image(image)

    with torch.no_grad():
        _, _, matte = modnet(im, True)

    alpha = postprocess_alpha(matte, original_size)
    processed_image = crop_above_neck(image, alpha)

    # Save the processed image in RGB format
    output_path = Path("app/static/uploads") / f"{file.filename}_processed.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))  # Keep colors intact

    return {"image_url": f"/static/uploads/{output_path.name}"}

@app.get("/download/{filename}", response_class=FileResponse)
async def download_image(filename: str):
    file_path = Path("app/static/uploads") / filename
    if file_path.exists():
        return file_path
    else:
        raise HTTPException(status_code=404, detail="File not found.")
