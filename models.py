from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from database import Base

class SensorData(Base):
    __tablename__ = 'sensor_data'

    id = Column(Integer, primary_key=True, index=True)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    image_data = Column(String, nullable=False)  

class AIPrediction(Base):
    __tablename__ = 'ai_predictions'

    id = Column(Integer, primary_key=True, index=True)
    image_data = Column(String, nullable=False)  
    predicted_class = Column(String, nullable=False) 
    confidence = Column(Float, nullable=False) 
