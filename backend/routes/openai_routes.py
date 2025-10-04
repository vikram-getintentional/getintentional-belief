from fastapi import APIRouter, Body, Depends
from backend.auth.auth_bearer import JWTBearer


router = APIRouter()
