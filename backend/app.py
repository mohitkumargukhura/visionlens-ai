from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
import io

# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI(
    title="VisionLens AI",
    description="Real-Time Image Classification API",
    version="1.0.0"
)

# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load Model Once
# --------------------------------------------------

MODEL_PATH = "models/keras_model.h5"
LABELS_PATH = "models/labels.txt"

model = load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r") as f:
    raw_labels = f.readlines()

labels = [line.strip()[2:] for line in raw_labels]

# --------------------------------------------------
# Request Model
# --------------------------------------------------

class PredictionRequest(BaseModel):
    image: str

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.asarray(image)

    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data = np.expand_dims(
        normalized_image_array,
        axis=0
    )

    return data

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "VisionLens AI",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "loaded",
        "classes": len(labels)
    }

@app.get("/labels")
def get_labels():
    return {
        "classes": labels
    }

@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        image_data = request.image

        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        data = preprocess_image(image_bytes)

        predictions = model.predict(
            data,
            verbose=0
        )[0]

        best_index = int(np.argmax(predictions))

        result_scores = []

        for i, score in enumerate(predictions):
            result_scores.append({
                "label": labels[i],
                "score": round(float(score * 100), 2)
            })

        result_scores.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "prediction": labels[best_index],
            "confidence": round(
                float(predictions[best_index] * 100),
                2
            ),
            "scores": result_scores
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )