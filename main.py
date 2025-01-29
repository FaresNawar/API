from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

class SensorData(BaseModel):
    temperature: float
    humidity: float

MODEL = tf.keras.models.load_model("C:/Users/zezoc/Desktop/Codes/PotatoCode/API/FloraiAPI/pythonapi/model.h5")
CLASS_NAMES= ["Early Blight","Late Blight","Healthy"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    
@app.post("/esp32/upload")
async def receive_from_esp32(
    sensor_data: SensorData, 
    image: bytes = File(...) 
):

    image_hex = image.hex()

    return JSONResponse(
        content={
            "message": "Image and sensor data received",
            "sensor_data": sensor_data.dict(),
            "image_bytes": image_hex
        }
    )
