import json
import asyncio
from typing import Dict, List
from fastapi import WebSocket
from datetime import datetime


class WebSocketManager:
    """
    Manages WebSocket connections for real-time notifications.
    """
    
    def __init__(self):
        # Store active connections by company_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, company_id: str):
        """Accept a WebSocket connection and add it to active connections."""
        await websocket.accept()
        
        if company_id not in self.active_connections:
            self.active_connections[company_id] = []
        
        self.active_connections[company_id].append(websocket)
        print(f"WebSocket connected for company_id: {company_id}")
    
    def disconnect(self, websocket: WebSocket, company_id: str):
        """Remove a WebSocket connection."""
        if company_id in self.active_connections:
            try:
                self.active_connections[company_id].remove(websocket)
                if not self.active_connections[company_id]:
                    del self.active_connections[company_id]
                print(f"WebSocket disconnected for company_id: {company_id}")
            except ValueError:
                pass  # Connection was not in the list
    
    async def send_personal_message(self, message: dict, company_id: str):
        """Send a message to all connections for a specific company."""
        if company_id in self.active_connections:
            message_str = json.dumps({
                **message,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Send to all connections for this company
            disconnected = []
            for connection in self.active_connections[company_id]:
                try:
                    await connection.send_text(message_str)
                except Exception as e:
                    print(f"Error sending WebSocket message: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.disconnect(connection, company_id)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all active connections."""
        message_str = json.dumps({
            **message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        for company_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, company_id)
    
    def get_connection_count(self, company_id: str = None) -> int:
        """Get the number of active connections."""
        if company_id:
            return len(self.active_connections.get(company_id, []))
        return sum(len(connections) for connections in self.active_connections.values())


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
