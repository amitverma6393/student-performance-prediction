"""
Model Evaluation and Diagnostic Visualizations Module
------------------------------------------------------
Computes standard regression metrics (MAE, RMSE, R-squared) and generates
publication-grade Plotly interactive figures for model diagnostics.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """
    Compute standard regression evaluation metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values from the model.

    Returns:
        Dict containing MAE, MSE, RMSE, and R2 rounded to 4 decimals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def get_metric_explanations() -> Dict[str, Dict[str, str]]:
    """
    Return beginner-friendly academic explanations of evaluation metrics
    specifically tailored for college presentations and viva discussions.
    """
    return {
        "MAE": {
            "name": "Mean Absolute Error",
            "formula": "MAE = (1/n) * Σ |y_actual - y_pred|",
            "explanation": (
                "The average magnitude of errors between predicted marks and actual marks, "
                "measured in the exact same units (marks out of 100). Lower values indicate better accuracy."
            ),
            "interpretation": "An MAE of 2.1 means predictions are, on average, within ±2.1 marks of the student's actual score.",
        },
        "RMSE": {
            "name": "Root Mean Squared Error",
            "formula": "RMSE = √ [ (1/n) * Σ (y_actual - y_pred)² ]",
            "explanation": (
                "Square root of average squared differences. Because errors are squared before averaging, "
                "RMSE penalizes large outliers more heavily than MAE does."
            ),
            "interpretation": "A lower RMSE confirms the model does not suffer from extreme wild predictions.",
        },
        "R2": {
            "name": "R-Squared (Coefficient of Determination)",
            "formula": "R² = 1 - (SS_res / SS_tot)",
            "explanation": (
                "Proportion of variance in the student's final marks that can be explained by the input features. "
                "Ranges from -∞ to 1.0 (where 1.0 represents perfect prediction)."
            ),
            "interpretation": "An R² of 0.88 means 88% of the variability in student marks is successfully captured by the model.",
        },
    }


def plot_actual_vs_predicted(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Trained Model",
) -> go.Figure:
    """Create interactive Plotly scatter plot of Actual vs Predicted marks."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    min_val = min(float(y_true_arr.min()), float(y_pred_arr.min())) - 2
    max_val = max(float(y_true_arr.max()), float(y_pred_arr.max())) + 2

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=y_true_arr,
            y=y_pred_arr,
            mode="markers",
            name="Test Predictions",
            marker=dict(
                color="#2563EB",
                size=7,
                opacity=0.7,
                line=dict(width=1, color="#1D4ED8"),
            ),
            hovertemplate="<b>Actual:</b> %{x:.1f} Marks<br><b>Predicted:</b> %{y:.1f} Marks<extra></extra>",
        )
    )

    # Ideal diagonal reference line (y = x)
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Fit (y = x)",
            line=dict(color="#DC2626", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Actual vs. Predicted Marks ({model_name})",
        xaxis_title="Actual Marks",
        yaxis_title="Predicted Marks",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(range=[min_val, max_val]),
        yaxis=dict(range=[min_val, max_val]),
    )
    return fig


def plot_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Trained Model",
) -> go.Figure:
    """Create interactive Plotly Residual plot (Residuals vs Predicted)."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    residuals = y_true_arr - y_pred_arr

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_pred_arr,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(
                color="#059669",
                size=7,
                opacity=0.7,
                line=dict(width=1, color="#047857"),
            ),
            hovertemplate="<b>Predicted:</b> %{x:.1f}<br><b>Residual:</b> %{y:.1f}<extra></extra>",
        )
    )

    # Zero line
    fig.add_hline(
        y=0,
        line=dict(color="#111827", width=2, dash="dash"),
        annotation_text="Zero Residual Line",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Residual Plot ({model_name})",
        xaxis_title="Predicted Marks",
        yaxis_title="Residual (Actual - Predicted)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_residual_distribution(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Trained Model",
) -> go.Figure:
    """Create interactive Plotly histogram showing residual distribution."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig = px.histogram(
        x=residuals,
        nbins=30,
        marginal="box",
        title=f"Residual Error Distribution ({model_name})",
        labels={"x": "Residual Error (Marks)"},
        color_discrete_sequence=["#6366F1"],
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_title="Residual Error (Actual - Predicted)",
        yaxis_title="Frequency",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_models_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Create interactive bar chart comparing metrics across candidate models."""
    fig = go.Figure()

    models = comparison_df["Model"].tolist()

    fig.add_trace(
        go.Bar(
            name="R² Score (Higher is better)",
            x=models,
            y=comparison_df["R2"],
            marker_color="#2563EB",
            text=[f"{v:.3f}" for v in comparison_df["R2"]],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="MAE (Lower is better)",
            x=models,
            y=comparison_df["MAE"],
            marker_color="#F59E0B",
            text=[f"{v:.2f}" for v in comparison_df["MAE"]],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="RMSE (Lower is better)",
            x=models,
            y=comparison_df["RMSE"],
            marker_color="#EF4444",
            text=[f"{v:.2f}" for v in comparison_df["RMSE"]],
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="group",
        title="Candidate Models Performance Comparison",
        xaxis_title="Model Algorithm",
        yaxis_title="Metric Value",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
