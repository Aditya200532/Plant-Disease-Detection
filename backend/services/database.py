from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient


class Database:
    def __init__(self):
        self.uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("MONGODB_DB", "plant_disease")
        self.client: AsyncIOMotorClient | None = None
        self.db = None

    async def connect(self):
        try:
            self.client = AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=3000)
            await self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            print(f"Connected to MongoDB: {self.db_name}")
        except Exception as exc:
            print(f"MongoDB unavailable ({exc}); prediction history will not be stored.")
            self.client = None
            self.db = None

    async def close(self):
        if self.client:
            self.client.close()

    @property
    def available(self) -> bool:
        return self.db is not None

    async def save_prediction(self, prediction: dict) -> str | None:
        if not self.available:
            return None
        doc = {
            **prediction,
            "timestamp": datetime.now(timezone.utc),
        }
        doc.pop("gradcam", None)
        result = await self.db.predictions.insert_one(doc)
        return str(result.inserted_id)

    async def get_history(self, limit: int = 50, skip: int = 0) -> list[dict]:
        if not self.available:
            return []
        cursor = self.db.predictions.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)

    async def get_stats(self) -> dict[str, Any]:
        if not self.available:
            return {"total": 0, "healthy": 0, "diseased": 0, "top_diseases": [], "db_available": False}
        total = await self.db.predictions.count_documents({})
        healthy = await self.db.predictions.count_documents({"status": "Healthy"})
        diseased = await self.db.predictions.count_documents({"status": "Diseased"})
        pipeline = [
            {"$match": {"status": "Diseased"}},
            {"$group": {"_id": {"plant": "$plant", "disease": "$disease"}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5},
        ]
        top_diseases = []
        async for doc in self.db.predictions.aggregate(pipeline):
            top_diseases.append({
                "plant": doc["_id"]["plant"],
                "disease": doc["_id"]["disease"],
                "count": doc["count"],
            })
        return {
            "total": total,
            "healthy": healthy,
            "diseased": diseased,
            "top_diseases": top_diseases,
            "db_available": True,
        }
