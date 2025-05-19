from fastapi.security import HTTPBearer
from fastapi import Request, HTTPException

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials = await super(JWTBearer, self).__call__(request)
        if credentials:
            if credentials.scheme != "Bearer":
                raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
            return credentials.credentials  # Just return the raw token
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")
