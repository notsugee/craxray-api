from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import cv2
from scipy.integrate import odeint
import base64
import matplotlib.pyplot as plt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('concrete_crack_detector_model.keras')

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((120, 120))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def calculate_crack_ratio(contour, image_shape):
    crack_area = cv2.contourArea(contour)
    total_area = image_shape[0] * image_shape[1]
    return crack_area / total_area

def estimate_crack_depth(gray_image, contour):
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mean_intensity = cv2.mean(gray_image, mask=mask)[0]
    max_depth = 10
    return (255 - mean_intensity) / 255 * max_depth

def predict_crack_growth(initial_crack_depth, num_days):
    C = 5e-12
    m = 3
    stress_range = 25
    Y = 1.12
    loading_cycles_per_day = 1500
    critical_crack_size_mm = 7.0

    def delta_K(a): return Y * stress_range * np.sqrt(np.pi * a)
    def crack_growth_rate(a, t): return C * (delta_K(a) ** m) * loading_cycles_per_day

    time_points = np.linspace(0, num_days, num_days + 1)
    crack_sizes = odeint(lambda a, t: crack_growth_rate(a, t), initial_crack_depth, time_points)
    final_crack_size = crack_sizes[-1, 0]
    
    critical_day = None
    for day, crack_size in zip(time_points, crack_sizes):
        if crack_size[0] >= critical_crack_size_mm:
            critical_day = int(day)
            break

    return final_crack_size, critical_day

@app.post("/predict/")
async def predict_crack(file: UploadFile = File(...), num_days: int = Form(30)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    result = "Crack detected" if prediction[0] >= 0.5 else "No crack detected"

    gray_image = cv2.cvtColor((image_array[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    fig, ax = plt.subplots()
    ax.imshow(gray_image, cmap='gray')
    ax.axis('off')
    gray_image_base64 = fig_to_base64(fig)

    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    fig, ax = plt.subplots()
    ax.imshow(binary_image, cmap='gray')
    ax.axis('off')
    binary_image_base64 = fig_to_base64(fig)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.copy(image_array[0])
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    fig, ax = plt.subplots()
    ax.imshow(contour_image)
    ax.axis('off')
    contour_image_base64 = fig_to_base64(fig)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image_array[0], (x, y), (x + w, y + h), (255, 0, 0), 2)

        fig, ax = plt.subplots()
        ax.imshow(image_array[0])
        ax.axis('off')
        bounding_box_image_base64 = fig_to_base64(fig)

    analysis = None

    if result == "Crack detected":
        crack_ratio = calculate_crack_ratio(largest_contour, image_array[0].shape)
        crack_depth = estimate_crack_depth(gray_image, largest_contour)

        final_crack_size, critical_day = predict_crack_growth(crack_depth, num_days)

        analysis = {
            "crack_ratio": crack_ratio,
            "estimated_depth_mm": round(crack_depth, 2),
            "predicted_growth_mm": round(final_crack_size, 2),
            "critical_day": critical_day
        }

    return {
        "result": result,
        "analysis": analysis,
        "data": {
            "grayscale_image": gray_image_base64,
            "binary_image": binary_image_base64,
            "contour_image": contour_image_base64,
            "bounding_box_image": bounding_box_image_base64
        }
    }

# To run the app, use: uvicorn app:app --reload
