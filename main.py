from fastapi import FastAPI, File, UploadFile, Form, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import SensorData, AIPrediction
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import json

app = FastAPI()

class SensorDataIn(BaseModel):
    temperature: float
    humidity: float

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/")
def read_root():
    return {"message": "Hello"}

@app.put("/predict")
async def predict(
    file: UploadFile = File(),
    db: Session = Depends(get_db)
):
    image_bytes = await file.read()
    image = read_file_as_image(image_bytes)
    image_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) 

    image_hex = file.filename.encode("utf-8").hex()

    ai_prediction = AIPrediction(
        image_data=image_hex,
        predicted_class=predicted_class,
        confidence=confidence
    )

    db.add(ai_prediction)
    db.commit()

    return {
        "image_filename": file.filename,
        "class": predicted_class,
        "confidence": confidence
    }

@app.put("/esp32/upload")
async def receive_from_esp32(
    sensor_data: str = Form(...),
    image: UploadFile = File(...),  
    db: Session = Depends(get_db)
):
    try:

        sensor_dict = json.loads(sensor_data)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"message": "Invalid JSON in sensor_data"})


    print(f"Received sensor data: {sensor_dict}")
    print(f"Received image data (first 100 bytes): {image.file.read(100)}")


    image_bytes = await image.read()
    image_hex = image_bytes.hex()


    db_sensor_data = SensorData(
        temperature=sensor_dict.get('temperature'),
        humidity=sensor_dict.get('humidity'),
        image_data=image_hex
    )

    db.add(db_sensor_data)
    db.commit()
    db.refresh(db_sensor_data)

    return JSONResponse(
        content={
            "message": "Image and sensor data received and stored",
            "sensor_data": sensor_dict,
            "image_bytes": image_hex
        }
    )
