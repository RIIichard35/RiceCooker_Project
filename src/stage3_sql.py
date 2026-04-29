from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime


class Stage3SQLRecorder:
    """
    Stage3 数据归档模块（SQLite）。

    作用：
    1) 为每个产品生成可追溯 product_id
    2) 记录每次检测的主结果（products）
    3) 记录帧级结果（frames，当前流程固定单帧）
    4) 记录 Stage2 缺陷明细（defects）
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    view TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    final_status TEXT NOT NULL,
                    fail_reason TEXT,
                    sharpest_idx INTEGER,
                    stage1_total_sec REAL,
                    stage2_sec REAL,
                    total_sec REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS frames (
                    frame_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    sharpness REAL,
                    stage1_status TEXT,
                    stage1_issues_json TEXT,
                    FOREIGN KEY(product_id) REFERENCES products(product_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    frame_idx INTEGER,
                    class_name TEXT NOT NULL,
                    score REAL,
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER,
                    is_consistent INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(product_id) REFERENCES products(product_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_products_status_created ON products(final_status, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_frames_product ON frames(product_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defects_product ON defects(product_id)"
            )
            conn.commit()

    def generate_product_id(self, shift: str = "A") -> str:
        """
        生成产品编号：
            P{YYYYMMDD}-{shift}-{seq:06d}
        例：
            P20260422-A-000123
        """
        date_str = datetime.now().strftime("%Y%m%d")
        prefix = f"P{date_str}-{shift}-"
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT product_id
                FROM products
                WHERE product_id LIKE ?
                ORDER BY product_id DESC
                LIMIT 1
                """,
                (f"{prefix}%",),
            ).fetchone()
        if row is None:
            seq = 1
        else:
            try:
                seq = int(row["product_id"].split("-")[-1]) + 1
            except Exception:
                seq = 1
        return f"{prefix}{seq:06d}"

    def save_inspection(
        self,
        *,
        product_id: str,
        view: str,
        frame_rows: list[dict],
        stage2_result: dict | None,
        final_status: str,
        fail_reason: str | None,
        inspected_idx: int | None,
        timing: dict | None,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        timing = timing or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO products (
                    product_id, view, created_at, final_status, fail_reason,
                    sharpest_idx, stage1_total_sec, stage2_sec, total_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    view,
                    now,
                    final_status,
                    fail_reason,
                    inspected_idx,
                    timing.get("stage1_total"),
                    timing.get("stage2"),
                    timing.get("total"),
                ),
            )

            conn.execute("DELETE FROM frames WHERE product_id = ?", (product_id,))
            for row in frame_rows:
                conn.execute(
                    """
                    INSERT INTO frames (
                        frame_id, product_id, frame_idx, sharpness,
                        stage1_status, stage1_issues_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["frame_id"],
                        product_id,
                        row["frame_idx"],
                        row.get("sharpness"),
                        row.get("stage1_status"),
                        json.dumps(row.get("stage1_issues", []), ensure_ascii=False),
                    ),
                )

            conn.execute("DELETE FROM defects WHERE product_id = ?", (product_id,))
            if stage2_result:
                for defect in stage2_result.get("defects", []):
                    x1, y1, x2, y2 = defect.get("xyxy", [None, None, None, None])
                    conn.execute(
                        """
                        INSERT INTO defects (
                            product_id, frame_idx, class_name, score, x1, y1, x2, y2, is_consistent
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            product_id,
                            inspected_idx,
                            defect.get("class_name", "unknown"),
                            defect.get("score"),
                            x1,
                            y1,
                            x2,
                            y2,
                            1,
                        ),
                    )
            conn.commit()

    def list_fail_products(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT product_id, view, created_at, final_status, fail_reason
                FROM products
                WHERE final_status = 'FAIL'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

