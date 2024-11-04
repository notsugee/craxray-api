from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import cv2
from scipy.integrate import odeint

# Create FastAPI instance
app = FastAPI()

# Allow all origins for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load the saved model
model = load_model('concrete_crack_detector_model.keras')

# Preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((120, 120))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Crack characteristic functions
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

# API route to upload an image and get a prediction
@app.post("/predict/")
async def predict_crack(file: UploadFile = File(...), num_days: int = Form(30)):
    # Read uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Preprocess the image and make prediction
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    result = "Crack detected" if prediction[0] >= 0.5 else "No crack detected"

    # Further analysis if crack detected
    if result == "Crack detected":
        image = image.resize((120, 120))
        image_np = np.array(image)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            crack_ratio = calculate_crack_ratio(largest_contour, image_np.shape)
            crack_depth = estimate_crack_depth(gray_image, largest_contour)

            # Predict crack growth
            final_crack_size, critical_day = predict_crack_growth(crack_depth, num_days)

            analysis = {
                "crack_ratio": crack_ratio,
                "estimated_depth_mm": crack_depth,
                "predicted_growth_mm": final_crack_size,
                "critical_day": critical_day
            }
        else:
            analysis = None
    else:
        analysis = None

    return {"result": result, "analysis": analysis}

# To run the app, use: uvicorn app:app --reload
