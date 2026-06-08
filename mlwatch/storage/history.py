import sqlite3
import json
import time
import numpy as np
from typing import List, Dict, Any


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class History:
    def __init__(self, db_path: str = "mlwatch.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    timestamp REAL,
                    result TEXT
                )
            """)
            self._conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT,
                        timestamp REAL,
                        result TEXT
                    )
                """)

    def _get_conn(self):
        if self.db_path == ":memory:":
            return self._conn
        return sqlite3.connect(self.db_path)

    def save(self, model_name: str, result: Dict[str, Any]):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO logs (model_name, timestamp, result) VALUES (?, ?, ?)",
            (model_name, time.time(), json.dumps(result, cls=NumpyEncoder)),
        )
        conn.commit()

    def get(self, model_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT timestamp, result FROM logs WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?",
            (model_name, limit),
        ).fetchall()
        return [{"timestamp": r[0], **json.loads(r[1])} for r in rows]
