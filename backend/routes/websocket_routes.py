from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from backend.utils.websocket_manager import websocket_manager
from backend.auth.jwt_handler import decode_token
import json


router = APIRouter()


@router.websocket("/ws/{company_id}")
async def websocket_endpoint(websocket: WebSocket, company_id: str, token: str = Query(...)):
    """
    WebSocket endpoint for real-time notifications.
    Requires authentication via token query parameter.
    """
    try:
        # Validate token
        decoded = decode_token(token)
        token_company_id = decoded.get("company_id")
        
        if not token_company_id or token_company_id != company_id:
            await websocket.close(code=4001, reason="Unauthorized")
            return
        
        # Connect the WebSocket
        await websocket_manager.connect(websocket, company_id)
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected to real-time notifications",
            "company_id": company_id
        }))
        
        try:
            # Keep the connection alive and handle incoming messages
            while True:
                # Wait for messages from client (like ping/pong)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle ping/pong for connection health
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "message": "Connection is alive"
                    }))
                
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket, company_id)
            print(f"WebSocket client disconnected for company_id: {company_id}")
            
    except Exception as e:
        print(f"WebSocket error for company_id {company_id}: {e}")
        try:
            await websocket.close(code=4000, reason="Internal server error")
        except:
            pass
