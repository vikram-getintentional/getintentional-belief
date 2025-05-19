# backend/auth/schemas.py
from pydantic import BaseModel

class UserCreate(BaseModel):
    email: str
    password: str
    company_name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    email: str
    company_name: str
    company_id: str

class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str
