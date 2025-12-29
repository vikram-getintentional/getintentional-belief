# backend/auth/auth_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi import Request, status
from fastapi.security import OAuth2PasswordBearer
from backend.auth.jwt_handler import decode_token
import uuid


from backend.database import SessionLocal
from backend.auth import models, schemas, hashing, jwt_handler
from backend.init_db import ensure_schema

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@router.get("/me", response_model=schemas.UserOut)
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = decode_token(token)
        email = payload.get("sub")
        user = db.query(models.User).filter(models.User.email == email).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except:
        raise HTTPException(status_code=401, detail="Invalid token")
    
@router.post("/signup", response_model=schemas.UserOut)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = hashing.hash_password(user.password)
    company_id = str(uuid.uuid4())
    new_user = models.User(
        email=user.email,
        hashed_password=hashed_pw,
        company_name=user.company_name,
        company_id=company_id
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    ensure_schema()
    return new_user

@router.post("/login", response_model=schemas.Token)
def login(user: schemas.LoginRequest, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not hashing.verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    ensure_schema()
    
    token = jwt_handler.create_access_token({
    "sub": db_user.email,
    "company_id": db_user.company_id  # ✅ Make sure this is included
})
    return {"access_token": token, "token_type": "bearer"}
