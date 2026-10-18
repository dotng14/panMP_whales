from __future__ import annotations
from fastapi import WebSocket


async def broadcast_json(clients: list[WebSocket], msg: dict) -> None:
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            clients.remove(ws)
