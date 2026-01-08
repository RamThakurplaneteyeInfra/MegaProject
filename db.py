from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 🔹 CHANGE THESE VALUES
DB_USER = "postgres"
DB_PASSWORD = "Admin"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "villages"

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
