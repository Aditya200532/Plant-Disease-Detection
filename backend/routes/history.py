from __future__ import annotations

from fastapi import APIRouter, Request, Query

router = APIRouter()


@router.get("/history")
async def get_history(request: Request, limit: int = Query(50, ge=1, le=200), skip: int = Query(0, ge=0)):
    db = request.app.state.db
    return await db.get_history(limit=limit, skip=skip)


@router.get("/stats")
async def get_stats(request: Request):
    db = request.app.state.db
    return await db.get_stats()
