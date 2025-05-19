from backend.database import Base, engine
from backend.auth import models

Base.metadata.create_all(bind=engine)