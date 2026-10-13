"""
Unit Tests for Data Ingestion, Preprocessing, and Validation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    create_preprocessor,
    load_data,
    split_data,
    validate_input_features,
)


@pytest.fixture
def sample_dataset_path(tmp_path):
    """Create a temporary valid dataset CSV."""
    file_path = tmp_path / "test_students.csv"
    data = {
        "Student_ID": [f"STD-{i}" for i in range(10)],
        "Attendance_Rate": [85.0, 70.0, 95.0, 60.0, 80.0, 90.0, 75.0, 88.0, 65.0, 92.0],
        "Study_Hours_Per_Day": [4.0, 2.5, 6.0, 1.5, 3.5, 5.0, 3.0, 4.5, 2.0, 5.5],
        "Previous_Exam_Marks": [75.0, 60.0, 90.0, 45.0, 70.0, 85.0, 65.0, 80.0, 55.0, 88.0],
        "Assignment_Score": [80.0, 65.0, 95.0, 50.0, 75.0, 90.0, 70.0, 85.0, 60.0, 92.0],
        "Sleep_Hours_Per_Day": [7.0, 6.5, 8.0, 5.5, 7.0, 7.5, 6.8, 7.2, 6.0, 8.0],
        "Parent_Education_Level": ["Bachelor", "High School", "Master", "High School", "Bachelor", "Doctorate", "Bachelor", "Master", "High School", "Bachelor"],
        "Internet_Access": ["Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "No", "Yes"],
        "Extracurricular_Activities": ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes"],
        "Final_Exam_Marks": [78.0, 62.0, 92.0, 48.0, 72.0, 87.0, 66.0, 82.0, 57.0, 90.0],
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path


def test_load_data_valid(sample_dataset_path):
    """Test loading a properly formatted dataset."""
    df = load_data(sample_dataset_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert TARGET_COLUMN in df.columns
    for col in ALL_FEATURE_COLUMNS:
        assert col in df.columns


def test_load_data_file_not_found():
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_directory/missing_file.csv")


def test_load_data_missing_columns(tmp_path):
    """Test that missing required columns raises ValueError."""
    bad_csv = tmp_path / "bad.csv"
    bad_df = pd.DataFrame({"Student_ID": ["STD-1"], "Attendance_Rate": [80.0]})
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError) as excinfo:
        load_data(bad_csv)
    assert "missing required columns" in str(excinfo.value)


def test_load_data_missing_values_imputed(tmp_path):
    """Test that null values are correctly imputed."""
    csv_with_nulls = tmp_path / "nulls.csv"
    data = {
        "Student_ID": [f"STD-{i}" for i in range(5)],
        "Attendance_Rate": [80.0, np.nan, 90.0, 70.0, 85.0],
        "Study_Hours_Per_Day": [3.0, 4.0, np.nan, 2.0, 5.0],
        "Previous_Exam_Marks": [70.0, 65.0, 85.0, np.nan, 80.0],
        "Assignment_Score": [75.0, 70.0, 90.0, 60.0, np.nan],
        "Sleep_Hours_Per_Day": [7.0, 7.5, 6.5, 8.0, np.nan],
        "Parent_Education_Level": ["Bachelor", None, "Master", "Bachelor", "Bachelor"],
        "Internet_Access": ["Yes", "Yes", None, "No", "Yes"],
        "Extracurricular_Activities": ["Yes", "No", "Yes", None, "Yes"],
        "Final_Exam_Marks": [75.0, 68.0, 88.0, 62.0, 82.0],
    }
    pd.DataFrame(data).to_csv(csv_with_nulls, index=False)
    cleaned_df = load_data(csv_with_nulls)

    assert not cleaned_df[ALL_FEATURE_COLUMNS].isnull().any().any()


def test_split_data(sample_dataset_path):
    """Test train-test splitting proportions and separation."""
    df = load_data(sample_dataset_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.3, random_state=42)

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3
    assert list(X_train.columns) == ALL_FEATURE_COLUMNS
    assert TARGET_COLUMN not in X_train.columns


def test_create_preprocessor_transformation(sample_dataset_path):
    """Test ColumnTransformer produces valid scaled/encoded numeric matrix."""
    df = load_data(sample_dataset_path)
    X_train, _, _, _ = split_data(df, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(X_train)
    # 5 numerical features + one-hot categories
    assert X_transformed.shape[1] >= len(NUMERICAL_FEATURES)


def test_validate_input_features_success():
    """Test valid input features return True with no errors."""
    valid_features = {
        "Attendance_Rate": 85.0,
        "Study_Hours_Per_Day": 4.0,
        "Previous_Exam_Marks": 72.0,
        "Assignment_Score": 80.0,
        "Sleep_Hours_Per_Day": 7.0,
        "Parent_Education_Level": "Bachelor",
        "Internet_Access": "Yes",
        "Extracurricular_Activities": "No",
    }
    is_valid, errors = validate_input_features(valid_features)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_input_features_out_of_bounds():
    """Test out-of-range numerical input is flagged."""
    invalid_features = {
        "Attendance_Rate": 150.0,  # Invalid: > 100
        "Study_Hours_Per_Day": -2.0,  # Invalid: < 0
        "Previous_Exam_Marks": 72.0,
        "Assignment_Score": 80.0,
        "Sleep_Hours_Per_Day": 7.0,
        "Parent_Education_Level": "InvalidDegree",  # Invalid category
        "Internet_Access": "Yes",
        "Extracurricular_Activities": "No",
    }
    is_valid, errors = validate_input_features(invalid_features)
    assert is_valid is False
    assert len(errors) == 3
