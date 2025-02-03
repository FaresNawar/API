from fastapi import FastAPI, File, UploadFile, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import SensorData, AIPrediction
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

class SensorDataIn(BaseModel):
    temperature: float
    humidity: float

MODEL = tf.keras.models.load_model("C:/Users/zezoc/Desktop/Codes/PotatoCode/API/FloraiAPI/pythonapi/model.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
import numpy as np


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Railway!"}

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
    sensor_data: SensorDataIn,
    image: bytes = File(...),
    db: Session = Depends(get_db)
):
    image_hex = image.hex()

    db_sensor_data = SensorData(
        temperature=sensor_data.temperature,
        humidity=sensor_data.humidity,
        image_data=image_hex 
    )
    
    db.add(db_sensor_data)
    db.commit()
    db.refresh(db_sensor_data)

    return JSONResponse(
        content={
            "message": "Image and sensor data received and stored",
            "sensor_data": sensor_data.dict(),
            "image_bytes": image_hex
        }
    )
