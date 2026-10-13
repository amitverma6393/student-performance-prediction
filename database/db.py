"""
SQLite Database Layer for Prediction History Management
-------------------------------------------------------
Provides lightweight, secure local storage for student prediction history,
auditing, date filtering, and CSV export.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd

DB_FILE_PATH = Path(__file__).resolve().parent / "predictions.db"


def get_connection() -> sqlite3.Connection:
    """Create and return a connection to the SQLite database."""
    DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the SQLite database and create schema if not already present."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                student_tag TEXT NOT NULL,
                attendance_rate REAL NOT NULL,
                study_hours REAL NOT NULL,
                prev_marks REAL NOT NULL,
                assignment_score REAL NOT NULL,
                sleep_hours REAL NOT NULL,
                parent_edu TEXT NOT NULL,
                internet_access TEXT NOT NULL,
                extracurricular TEXT NOT NULL,
                predicted_marks REAL NOT NULL,
                performance_category TEXT NOT NULL,
                model_used TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_prediction(
    features: Dict[str, Any],
    predicted_marks: float,
    performance_category: str,
    student_tag: str = "Anonymous",
    model_used: str = "Linear Regression Pipeline",
) -> int:
    """
    Save a student performance prediction record to the SQLite database.

    Returns:
        int: The inserted record ID.
    """
    init_db()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_history (
                timestamp, student_tag, attendance_rate, study_hours,
                prev_marks, assignment_score, sleep_hours, parent_edu,
                internet_access, extracurricular, predicted_marks,
                performance_category, model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str,
                student_tag,
                float(features.get("Attendance_Rate", 0.0)),
                float(features.get("Study_Hours_Per_Day", 0.0)),
                float(features.get("Previous_Exam_Marks", 0.0)),
                float(features.get("Assignment_Score", 0.0)),
                float(features.get("Sleep_Hours_Per_Day", 0.0)),
                str(features.get("Parent_Education_Level", "Unknown")),
                str(features.get("Internet_Access", "Unknown")),
                str(features.get("Extracurricular_Activities", "Unknown")),
                round(float(predicted_marks), 2),
                performance_category,
                model_used,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_all_predictions(
    category_filter: Optional[str] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch prediction history records as a Pandas DataFrame.

    Args:
        category_filter: Optional category to filter by (e.g. 'Excellent').
        limit: Maximum number of rows to retrieve.
    """
    init_db()
    query = "SELECT * FROM prediction_history"
    params: List[Any] = []

    if category_filter and category_filter != "All":
        query += " WHERE performance_category = ?"
        params.append(category_filter)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df


def delete_prediction(record_id: int) -> bool:
    """Delete a single prediction record by ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_predictions() -> bool:
    """Delete all records from the prediction history table."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history")
        conn.commit()
        return True
