"""
Unit Tests for Prediction Inference, Categorization, and SQLite Database
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db import (
    clear_all_predictions,
    delete_prediction,
    get_all_predictions,
    init_db,
    save_prediction,
)
from src.prediction import (
    categorize_performance,
    generate_student_insights,
    load_model_metadata,
    load_model_pipeline,
    predict_student_performance,
)


@pytest.fixture
def sample_valid_features():
    return {
        "Attendance_Rate": 85.0,
        "Study_Hours_Per_Day": 4.5,
        "Previous_Exam_Marks": 75.0,
        "Assignment_Score": 80.0,
        "Sleep_Hours_Per_Day": 7.5,
        "Parent_Education_Level": "Bachelor",
        "Internet_Access": "Yes",
        "Extracurricular_Activities": "Yes",
    }


def test_load_pipeline_and_metadata():
    """Verify that saved pipeline and metadata can be loaded."""
    pipeline = load_model_pipeline()
    assert pipeline is not None
    assert hasattr(pipeline, "predict")

    metadata = load_model_metadata()
    assert isinstance(metadata, dict)
    assert "selected_model" in metadata
    assert "best_metrics" in metadata
    assert "R2" in metadata["best_metrics"]


def test_predict_performance_valid_range(sample_valid_features):
    """Verify prediction output range is bounded within 0 to 100."""
    result = predict_student_performance(sample_valid_features)
    assert "predicted_marks" in result
    marks = result["predicted_marks"]
    assert isinstance(marks, float)
    assert 0.0 <= marks <= 100.0

    assert "performance_tier" in result
    assert result["performance_tier"] in ["Excellent", "Good", "Average", "Needs Improvement"]
    assert "insights" in result
    assert isinstance(result["insights"], list)


def test_performance_categorization_thresholds():
    """Verify accurate academic grading tier classifications."""
    assert categorize_performance(92.0)[0] == "Excellent"
    assert categorize_performance(85.0)[0] == "Excellent"
    assert categorize_performance(78.5)[0] == "Good"
    assert categorize_performance(70.0)[0] == "Good"
    assert categorize_performance(62.0)[0] == "Average"
    assert categorize_performance(50.0)[0] == "Average"
    assert categorize_performance(45.0)[0] == "Needs Improvement"
    assert categorize_performance(25.0)[0] == "Needs Improvement"


def test_prediction_invalid_feature_raises():
    """Verify invalid input features raise an exception."""
    bad_features = {
        "Attendance_Rate": 120.0,  # Invalid
        "Study_Hours_Per_Day": 4.0,
        "Previous_Exam_Marks": 70.0,
        "Assignment_Score": 80.0,
        "Sleep_Hours_Per_Day": 7.0,
        "Parent_Education_Level": "Bachelor",
        "Internet_Access": "Yes",
        "Extracurricular_Activities": "Yes",
    }
    with pytest.raises(ValueError):
        predict_student_performance(bad_features)


def test_student_insights_triggers():
    """Verify that specific behavioral thresholds produce targeted advice."""
    at_risk_features = {
        "Attendance_Rate": 55.0,  # Below 75%
        "Study_Hours_Per_Day": 1.0,  # Below 2.0 hrs
        "Previous_Exam_Marks": 50.0,
        "Assignment_Score": 50.0,  # Below 60%
        "Sleep_Hours_Per_Day": 4.5,  # Below 6.0 hrs
        "Parent_Education_Level": "High School",
        "Internet_Access": "Yes",
        "Extracurricular_Activities": "No",
    }
    insights = generate_student_insights(at_risk_features, predicted_marks=45.0)
    titles = [ins["title"] for ins in insights]

    assert any("Attendance" in t for t in titles)
    assert any("Study" in t for t in titles)
    assert any("Sleep" in t for t in titles)


def test_sqlite_database_lifecycle(sample_valid_features):
    """Verify SQLite database initialization, record insertion, querying, and deletion."""
    init_db()

    # Clear existing test records
    clear_all_predictions()
    empty_df = get_all_predictions()
    assert len(empty_df) == 0

    # Insert a record
    inserted_id = save_prediction(
        features=sample_valid_features,
        predicted_marks=78.5,
        performance_category="Good",
        student_tag="Test-Unit-101",
        model_used="Test Model Pipeline",
    )
    assert inserted_id > 0

    # Query records
    history_df = get_all_predictions()
    assert len(history_df) == 1
    assert history_df.iloc[0]["student_tag"] == "Test-Unit-101"
    assert history_df.iloc[0]["predicted_marks"] == 78.5
    assert history_df.iloc[0]["performance_category"] == "Good"

    # Delete record
    deleted = delete_prediction(inserted_id)
    assert deleted is True

    # Confirm table is empty again
    after_delete_df = get_all_predictions()
    assert len(after_delete_df) == 0
