"""
Machine Learning Training Pipeline
----------------------------------
Trains, evaluates, and compares multiple regression models
(Linear Regression, Decision Tree, Random Forest).
Selects the best performing model based on R-squared & RMSE,
and persists the final Scikit-learn Pipeline and metadata.
"""

import sys
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data_preprocessing import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    create_preprocessor,
    load_data,
    split_data,
)
from src.evaluation import compute_regression_metrics


def build_candidate_models(preprocessor) -> Dict[str, Pipeline]:
    """Build candidate Scikit-learn Pipelines combining preprocessor and regressors."""
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        ),
        "Decision Tree Regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    DecisionTreeRegressor(
                        max_depth=6,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_split=6,
                        min_samples_leaf=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def extract_feature_importance(pipeline: Pipeline) -> Dict[str, float]:
    """Extract feature importance or magnitude coefficients for interpretability."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        regressor = pipeline.named_steps["regressor"]

        # Retrieve feature names from ColumnTransformer
        num_cols = NUMERICAL_FEATURES
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_cols = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        all_cols = num_cols + cat_cols

        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            return {col: round(float(imp), 4) for col, imp in zip(all_cols, importances)}
        elif hasattr(regressor, "coef_"):
            coefs = np.abs(regressor.coef_)
            total = np.sum(coefs)
            rel_importance = coefs / total if total > 0 else coefs
            return {col: round(float(imp), 4) for col, imp in zip(all_cols, rel_importance)}
    except Exception as exc:
        print(f"Notice: Could not extract feature importances: {exc}")
    return {}


def train_and_save_pipeline(
    data_path: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Execute full training workflow:
    1. Load and split dataset
    2. Train all candidate models
    3. Evaluate test metrics
    4. Pick the winner
    5. Save model pipeline (.joblib) and metadata (.json)
    """
    root_dir = Path(__file__).resolve().parent.parent

    if data_path is None:
        data_path = root_dir / "data" / "student_performance.csv"
    else:
        data_path = Path(data_path)

    if models_dir is None:
        models_dir = root_dir / "models"
    else:
        models_dir = Path(models_dir)

    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_path.resolve()}")
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    preprocessor = create_preprocessor()
    candidate_models = build_candidate_models(preprocessor)

    results = []
    trained_pipelines = {}
    test_predictions_dict = {}

    print("\nTraining and evaluating candidate models...")
    for model_name, pipeline in candidate_models.items():
        print(f"  Training {model_name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = compute_regression_metrics(y_test, y_pred)
        metrics["Model"] = model_name
        results.append(metrics)
        trained_pipelines[model_name] = pipeline
        test_predictions_dict[model_name] = [round(float(v), 2) for v in y_pred]

        print(f"    R²: {metrics['R2']:.4f} | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f}")

    results_df = pd.DataFrame(results)

    # Select best model based on highest R2 (and lowest RMSE as tie breaker)
    best_row = results_df.sort_values(by=["R2", "RMSE"], ascending=[False, True]).iloc[0]
    best_model_name = str(best_row["Model"])
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\n=> Best Model Selected: {best_model_name} (R² = {best_row['R2']:.4f})")

    # Save best pipeline
    pipeline_save_path = models_dir / "student_performance_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_save_path)
    print(f"Saved trained pipeline to: {pipeline_save_path.resolve()}")

    # Also save all candidate pipelines for interactive evaluation comparison
    all_models_path = models_dir / "candidate_models.joblib"
    joblib.dump(trained_pipelines, all_models_path)
    print(f"Saved candidate models to: {all_models_path.resolve()}")

    # Feature importances
    feature_importances = extract_feature_importance(best_pipeline)

    # Prepare metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "selected_model": best_model_name,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "target_column": TARGET_COLUMN,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_feature_columns": ALL_FEATURE_COLUMNS,
        "best_metrics": {
            "R2": float(best_row["R2"]),
            "MAE": float(best_row["MAE"]),
            "RMSE": float(best_row["RMSE"]),
            "MSE": float(best_row["MSE"]),
        },
        "all_model_comparison": results_df.to_dict(orient="records"),
        "feature_importances": feature_importances,
        "test_actuals": [round(float(v), 2) for v in y_test.tolist()],
        "test_predictions": test_predictions_dict,
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {metadata_path.resolve()}")

    return metadata


def main():
    print("=" * 60)
    print("  STUDENT PERFORMANCE PREDICTION - MODEL TRAINING RUN")
    print("=" * 60)
    train_and_save_pipeline()
    print("=" * 60)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
