import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env file (useful for local testing)
load_dotenv()

# Get DATABASE_URL from Railway or fallback to local PostgreSQL
DATABASE_URL = "postgresql://postgres:TisqSvFqDyAVcXQtgZUstpSXRwarxxNq@postgres.railway.internal:5432/railway"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
