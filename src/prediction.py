"""
Inference, Categorization, and Student Insights Module
------------------------------------------------------
Loads the persistent Scikit-learn Pipeline, validates inference inputs,
computes score predictions, maps to standard academic performance tiers,
and generates actionable, non-judgmental guidance.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    ALL_FEATURE_COLUMNS,
    validate_input_features,
)

# Standard Academic Thresholds for College Grading
PERFORMANCE_TIERS = {
    "Excellent": {"min": 85.0, "max": 100.0, "color": "#10B981", "badge": "success"},
    "Good": {"min": 70.0, "max": 84.99, "color": "#2563EB", "badge": "info"},
    "Average": {"min": 50.0, "max": 69.99, "color": "#F59E0B", "badge": "warning"},
    "Needs Improvement": {"min": 0.0, "max": 49.99, "color": "#EF4444", "badge": "error"},
}


def load_model_pipeline(pipeline_path: str | Path | None = None):
    """Load the serialized Scikit-learn pipeline from disk."""
    if pipeline_path is None:
        root_dir = Path(__file__).resolve().parent.parent
        pipeline_path = root_dir / "models" / "student_performance_pipeline.joblib"
    else:
        pipeline_path = Path(pipeline_path)

    if not pipeline_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {pipeline_path.resolve()}. "
            "Please run `python src/train_model.py` first."
        )

    return joblib.load(pipeline_path)


def load_model_metadata(metadata_path: str | Path | None = None) -> Dict[str, Any]:
    """Load model training metadata and performance statistics."""
    if metadata_path is None:
        root_dir = Path(__file__).resolve().parent.parent
        metadata_path = root_dir / "models" / "model_metadata.json"
    else:
        metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def categorize_performance(marks: float) -> Tuple[str, str, str]:
    """
    Map numerical marks to academic tier, color hex, and Streamlit badge type.

    Returns:
        Tuple of (tier_name, color_hex, badge_type).
    """
    marks = float(marks)
    for tier, conf in PERFORMANCE_TIERS.items():
        if conf["min"] <= marks <= conf["max"]:
            return tier, conf["color"], conf["badge"]
    if marks > 100.0:
        return "Excellent", "#10B981", "success"
    return "Needs Improvement", "#EF4444", "error"


def generate_student_insights(
    features: Dict[str, Any],
    predicted_marks: float,
) -> List[Dict[str, str]]:
    """
    Generate constructive, non-judgmental, actionable recommendations
    based on the student's study habits and input indicators.
    """
    insights: List[Dict[str, str]] = []

    att = float(features.get("Attendance_Rate", 0.0))
    study = float(features.get("Study_Hours_Per_Day", 0.0))
    prev = float(features.get("Previous_Exam_Marks", 0.0))
    assign = float(features.get("Assignment_Score", 0.0))
    sleep = float(features.get("Sleep_Hours_Per_Day", 0.0))

    # Attendance guidance
    if att < 75.0:
        insights.append({
            "category": "Attendance Alert",
            "type": "warning",
            "title": "Low Attendance Risk (< 75%)",
            "detail": (
                f"Current attendance is {att:.1f}%. Most universities mandate at least 75% "
                "to qualify for semester examinations. Consistent lecture presence strongly correlates "
                "with understanding foundational concepts."
            ),
        })
    else:
        insights.append({
            "category": "Attendance Commendation",
            "type": "success",
            "title": "Healthy Attendance Rate",
            "detail": (
                f"Your attendance of {att:.1f}% satisfies university eligibility norms "
                "and provides a solid foundation for classroom engagement."
            ),
        })

    # Study hours guidance
    if study < 2.0:
        insights.append({
            "category": "Study Habits",
            "type": "warning",
            "title": "Recommended Self-Study Increase",
            "detail": (
                f"Allocating {study:.1f} hours/day is on the lower side for engineering coursework. "
                "Consider incremental 30-minute study blocks focused on active recall and problem-solving."
            ),
        })
    elif study > 6.0:
        insights.append({
            "category": "Study Habits",
            "type": "info",
            "title": "High Study Volume - Monitor Burnout",
            "detail": (
                f"You study {study:.1f} hours/day. While dedication is commendable, ensure "
                "you incorporate spaced intervals (Pomodoro technique) to prevent cognitive fatigue."
            ),
        })
    else:
        insights.append({
            "category": "Study Habits",
            "type": "success",
            "title": "Balanced Study Routine",
            "detail": f"{study:.1f} hours of daily dedicated study is a healthy, sustainable study window.",
        })

    # Assignment score guidance
    if assign < 60.0:
        insights.append({
            "category": "Internal Assessments",
            "type": "warning",
            "title": "Improvement in Coursework Recommended",
            "detail": (
                f"Assignment score is {assign:.1f}%. Strengthening lab submissions and internal assignments "
                "boosts continuous evaluation marks and solidifies test preparedness."
            ),
        })

    # Sleep & Cognitive Well-being
    if sleep < 6.0:
        insights.append({
            "category": "Well-Being",
            "type": "warning",
            "title": "Sleep Deprivation Risk",
            "detail": (
                f"Reporting {sleep:.1f} hours of sleep. Research in cognitive psychology shows that "
                "chronic sleep deprivation severely degrades memory consolidation and examination focus."
            ),
        })
    elif 6.5 <= sleep <= 8.5:
        insights.append({
            "category": "Well-Being",
            "type": "success",
            "title": "Optimal Sleep Schedule",
            "detail": f"Getting {sleep:.1f} hours of rest promotes effective long-term memory retention.",
        })

    # Academic trajectory comparison
    diff = predicted_marks - prev
    if diff > 5.0:
        insights.append({
            "category": "Progress Trajectory",
            "type": "success",
            "title": "Positive Growth Trajectory",
            "detail": (
                f"Projected performance ({predicted_marks:.1f}) shows a potential +{diff:.1f} gain "
                f"over your previous exam marks ({prev:.1f})."
            ),
        })
    elif diff < -5.0:
        insights.append({
            "category": "Progress Trajectory",
            "type": "warning",
            "title": "Dip Relative to Previous Score",
            "detail": (
                f"Projected marks ({predicted_marks:.1f}) fall below your previous exam marks ({prev:.1f}). "
                "Revisiting study regularity or assignment submission can help bridge this gap."
            ),
        })

    return insights


def predict_student_performance(
    features: Dict[str, Any],
    pipeline=None,
) -> Dict[str, Any]:
    """
    Validate inputs, perform prediction using the pipeline, and return
    predicted marks, tier categorization, and actionable student insights.

    Returns:
        Dict containing prediction results and diagnostic metadata.
    """
    # 1. Validate inputs
    is_valid, errors = validate_input_features(features)
    if not is_valid:
        raise ValueError("; ".join(errors))

    # 2. Ensure pipeline is loaded
    if pipeline is None:
        pipeline = load_model_pipeline()

    # 3. Create DataFrame with ordered features
    input_row = {col: features[col] for col in ALL_FEATURE_COLUMNS}
    input_df = pd.DataFrame([input_row])

    # 4. Predict
    raw_pred = pipeline.predict(input_df)[0]
    predicted_marks = round(float(np.clip(raw_pred, 0.0, 100.0)), 1)

    # 5. Categorize
    tier, color, badge = categorize_performance(predicted_marks)

    # 6. Generate Insights
    insights = generate_student_insights(features, predicted_marks)

    return {
        "predicted_marks": predicted_marks,
        "performance_tier": tier,
        "tier_color": color,
        "tier_badge": badge,
        "input_features": features,
        "insights": insights,
    }
