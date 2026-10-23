import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, String, Float, DateTime, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or "sqlite:///./powerpredict.db"
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    target_date: Mapped[str] = mapped_column(String(20))
    temperature: Mapped[float] = mapped_column(Float)
    humidity: Mapped[float] = mapped_column(Float)
    day_type: Mapped[str] = mapped_column(String(30))
    previous_consumption: Mapped[float] = mapped_column(Float)
    predicted_kwh: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=80.0)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
