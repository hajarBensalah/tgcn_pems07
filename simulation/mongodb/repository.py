"""
simulation/mongodb/repository.py
--------------------------------
Couche d'accès MongoDB pour stocker les données de simulation SUMO.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from simulation.config import (
    BOULEVARDS,
    COLLECTIONS,
    MONGO_DB,
    MONGO_URI,
    TEST_RATIO,
    TGCN_RATIO,
)


class MongoRepository:
    """Gestionnaire MongoDB pour la simulation Casablanca."""

    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db: Database = self.client[db_name]
        self._ensure_indexes()

    def _col(self, name: str) -> Collection:
        return self.db[COLLECTIONS[name]]

    def _ensure_indexes(self):
        ts = self._col("timeseries")
        ts.create_index([("run_id", ASCENDING), ("step", ASCENDING)])
        ts.create_index([("run_id", ASCENDING), ("boulevard_id", ASCENDING)])
        ts.create_index([("split", ASCENDING)])

        metrics = self._col("metrics")
        metrics.create_index([("run_id", ASCENDING), ("boulevard_id", ASCENDING)])

    def ping(self) -> bool:
        self.client.admin.command("ping")
        return True

    def init_boulevards(self):
        """Insère ou met à jour la définition des 5 boulevards."""
        col = self._col("boulevards")
        for b in BOULEVARDS:
            col.update_one(
                {"id": b["id"]},
                {"$set": {**b, "updated_at": datetime.now(timezone.utc)}},
                upsert=True,
            )

    def create_run(self, config: dict) -> str:
        run_id = str(uuid4())
        doc = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc),
            "status": "running",
            "config": config,
            "num_boulevards": len(BOULEVARDS),
        }
        self._col("runs").insert_one(doc)
        return run_id

    def finish_run(self, run_id: str, summary: dict):
        self._col("runs").update_one(
            {"run_id": run_id},
            {"$set": {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc),
                "summary": summary,
            }},
        )

    def insert_timeseries_batch(self, records: list[dict]):
        if records:
            self._col("timeseries").insert_many(records)

    def assign_splits(self, run_id: str, test_ratio: float = TEST_RATIO):
        """
        Assigne chaque pas de temps à 'test' ou 'tgcn'.
        Split temporel : premiers (1-test_ratio) → test, reste → tgcn.
        """
        col = self._col("timeseries")
        steps = sorted(col.distinct("step", {"run_id": run_id}))
        if not steps:
            return

        split_idx = int(len(steps) * test_ratio)
        test_steps = set(steps[:split_idx])
        tgcn_steps = set(steps[split_idx:])

        for step in test_steps:
            col.update_many(
                {"run_id": run_id, "step": step},
                {"$set": {"split": "test"}},
            )
        for step in tgcn_steps:
            col.update_many(
                {"run_id": run_id, "step": step},
                {"$set": {"split": "tgcn"}},
            )

        self._col("splits").insert_one({
            "run_id": run_id,
            "test_steps": len(test_steps),
            "tgcn_steps": len(tgcn_steps),
            "test_ratio": test_ratio,
            "tgcn_ratio": TGCN_RATIO,
            "created_at": datetime.now(timezone.utc),
        })

    def store_boulevard_metrics(self, run_id: str, metrics: list[dict]):
        for m in metrics:
            m["run_id"] = run_id
            m["created_at"] = datetime.now(timezone.utc)
        if metrics:
            self._col("metrics").insert_many(metrics)

    def get_timeseries_matrix(self, run_id: str, split: str = "tgcn") -> dict:
        """
        Retourne une matrice [T, N] des vitesses moyennes par boulevard.
        """
        import numpy as np

        col = self._col("timeseries")
        query = {"run_id": run_id}
        if split:
            query["split"] = split

        records = list(col.find(query).sort([("step", ASCENDING), ("boulevard_id", ASCENDING)]))
        if not records:
            return {"data": None, "steps": [], "boulevard_ids": []}

        steps = sorted(set(r["step"] for r in records))
        boulevard_ids = sorted(set(r["boulevard_id"] for r in records))
        T, N = len(steps), len(boulevard_ids)

        matrix = np.full((T, N), np.nan, dtype=np.float32)
        step_idx = {s: i for i, s in enumerate(steps)}
        blvd_idx = {b: i for i, b in enumerate(boulevard_ids)}

        for r in records:
            matrix[step_idx[r["step"]], blvd_idx[r["boulevard_id"]]] = r["avg_speed_kmh"]

        return {
            "data": matrix,
            "steps": steps,
            "boulevard_ids": boulevard_ids,
        }

    def get_metrics_by_boulevard(self, run_id: str) -> list[dict]:
        return list(self._col("metrics").find({"run_id": run_id}))

    def get_latest_run(self) -> dict | None:
        return self._col("runs").find_one(
            {"status": "completed"},
            sort=[("created_at", -1)],
        )

    def close(self):
        self.client.close()
