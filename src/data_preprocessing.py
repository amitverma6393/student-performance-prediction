"""
Data Preprocessing Module
-------------------------
Handles dataset ingestion, schema validation, missing value imputation,
feature separation, train-test splitting, and Scikit-learn ColumnTransformer
pipeline construction.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical Feature Definitions
NUMERICAL_FEATURES: List[str] = [
    "Attendance_Rate",
    "Study_Hours_Per_Day",
    "Previous_Exam_Marks",
    "Assignment_Score",
    "Sleep_Hours_Per_Day",
]

CATEGORICAL_FEATURES: List[str] = [
    "Parent_Education_Level",
    "Internet_Access",
    "Extracurricular_Activities",
]

ALL_FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "Final_Exam_Marks"

# Allowed Ranges and Categories for Validation
FEATURE_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "Attendance_Rate": {"min": 0.0, "max": 100.0, "type": float},
    "Study_Hours_Per_Day": {"min": 0.0, "max": 24.0, "type": float},
    "Previous_Exam_Marks": {"min": 0.0, "max": 100.0, "type": float},
    "Assignment_Score": {"min": 0.0, "max": 100.0, "type": float},
    "Sleep_Hours_Per_Day": {"min": 0.0, "max": 24.0, "type": float},
    "Parent_Education_Level": {
        "categories": ["High School", "Bachelor", "Master", "Doctorate"],
        "type": str,
    },
    "Internet_Access": {"categories": ["Yes", "No"], "type": str},
    "Extracurricular_Activities": {"categories": ["Yes", "No"], "type": str},
}


def get_feature_definitions() -> Dict[str, Any]:
    """Return dictionary of feature definitions and constraints."""
    return {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "constraints": FEATURE_CONSTRAINTS,
    }


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load dataset from a CSV path and validate required columns.

    Args:
        filepath: Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    df = pd.read_csv(path)

    # Check for required columns
    required_cols = set(ALL_FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(list(missing_cols))}")

    # Handle missing values if any exist
    if df[ALL_FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        # Fill numerical columns with median, categorical with mode
        for col in NUMERICAL_FEATURES:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        for col in CATEGORICAL_FEATURES:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        if df[TARGET_COLUMN].isnull().any():
            df = df.dropna(subset=[TARGET_COLUMN])

    # Remove any potential duplicates in student records
    if "Student_ID" in df.columns:
        df = df.drop_duplicates(subset=["Student_ID"])
    else:
        df = df.drop_duplicates()

    return df


def create_preprocessor() -> ColumnTransformer:
    """
    Create a Scikit-learn ColumnTransformer that scales numerical features
    and one-hot encodes categorical features.

    Returns:
        ColumnTransformer: Preprocessing transformer ready for Pipeline integration.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing feature/target sets.

    Args:
        df: Input DataFrame containing features and target.
        test_size: Fraction of samples for the test set (default: 0.2).
        random_state: Random state for reproducibility (default: 42).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df[ALL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def validate_input_features(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate an input feature dictionary for single student prediction.

    Args:
        features: Dictionary mapping feature column names to input values.

    Returns:
        Tuple of (is_valid: bool, errors: List[str]).
    """
    errors: List[str] = []

    for col in ALL_FEATURE_COLUMNS:
        if col not in features:
            errors.append(f"Missing required feature: '{col}'")
            continue

        val = features[col]
        constraints = FEATURE_CONSTRAINTS[col]

        # Numerical checks
        if col in NUMERICAL_FEATURES:
            try:
                num_val = float(val)
                min_val = constraints["min"]
                max_val = constraints["max"]
                if not (min_val <= num_val <= max_val):
                    errors.append(
                        f"'{col}' must be between {min_val} and {max_val}. Received: {num_val}"
                    )
            except (ValueError, TypeError):
                errors.append(f"'{col}' must be a numerical value. Received: {val}")

        # Categorical checks
        elif col in CATEGORICAL_FEATURES:
            allowed = constraints["categories"]
            if str(val) not in allowed:
                errors.append(
                    f"'{col}' must be one of {allowed}. Received: '{val}'"
                )

    return len(errors) == 0, errors
