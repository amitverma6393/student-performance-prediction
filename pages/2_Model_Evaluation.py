"""
Module 4 & 5: Model Evaluation & Performance Comparison
-------------------------------------------------------
Presents cross-model benchmarking (Linear Regression, Decision Tree, Random Forest),
regression diagnostics, residual analysis, and viva viva metric explanations.
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    get_metric_explanations,
    plot_actual_vs_predicted,
    plot_models_comparison,
    plot_residual_distribution,
    plot_residuals,
)
from src.prediction import load_model_metadata
from src.train_model import train_and_save_pipeline

st.set_page_config(
    page_title="Model Evaluation & Comparison",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Machine Learning Model Evaluation & Comparison")
st.caption("Benchmark candidate regression models, evaluate test errors, and analyze model goodness-of-fit.")

metadata = load_model_metadata()

if not metadata:
    st.warning("⚠️ Model metadata not found. Training models now...")
    with st.spinner("Training models and computing metrics..."):
        metadata = train_and_save_pipeline()
    st.success("Training complete! Refreshing page...")
    st.rerun()

# Retrain Model Button in Expander
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Control")
    st.caption("Trigger full training cycle across all 3 models.")
    if st.button("🔄 Retrain All Models Now", use_container_width=True):
        with st.spinner("Executing training pipeline..."):
            metadata = train_and_save_pipeline()
        st.success("Models retrained and re-evaluated!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 Best Pipeline")
    st.info(f"**Model:** {metadata.get('selected_model')}\n\n**Test R²:** {metadata.get('best_metrics', {}).get('R2'):.4f}\n\n**Test RMSE:** {metadata.get('best_metrics', {}).get('RMSE'):.2f}")

# Top Cards for Best Model
best_metrics = metadata.get("best_metrics", {})
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🏆 Best Algorithm", metadata.get("selected_model", "N/A"))
with c2:
    st.metric("R² Score (Variance Explained)", f"{best_metrics.get('R2', 0.0):.4f}", help="Higher is better (Max 1.0)")
with c3:
    st.metric("Mean Absolute Error (MAE)", f"{best_metrics.get('MAE', 0.0):.2f} Marks", help="Average prediction error in marks")
with c4:
    st.metric("Root Mean Squared Error (RMSE)", f"{best_metrics.get('RMSE', 0.0):.2f} Marks", help="Penalizes large errors")

st.markdown("---")

# Main Section: Tabs
tab_comparison, tab_diagnostics, tab_features, tab_viva = st.tabs([
    "📊 Cross-Model Comparison Table",
    "🎯 Regression Diagnostics (Actual vs Pred & Residuals)",
    "🌲 Feature Importance",
    "📚 Metric Glossary & Viva Guide",
])

# -------------------------------------------------------------
# TAB 1: MODEL COMPARISON
# -------------------------------------------------------------
with tab_comparison:
    st.subheader("Benchmark Across Candidate Regression Models")
    st.markdown(
        "Each model was evaluated on the identical held-out test dataset (300 student records) "
        "following identical pre-processing via Scikit-learn Pipeline."
    )

    comparison_records = metadata.get("all_model_comparison", [])
    if comparison_records:
        comp_df = pd.DataFrame(comparison_records)
        # Format columns
        st.dataframe(
            comp_df[["Model", "R2", "MAE", "RMSE", "MSE"]].style.highlight_max(subset=["R2"], color="#BBF7D0")
            .highlight_min(subset=["MAE", "RMSE"], color="#BBF7D0"),
            use_container_width=True,
        )

        st.markdown("#### Comparative Visualization")
        fig_comp = plot_models_comparison(comp_df)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.info(
            f"**Selection Conclusion:** **{metadata.get('selected_model')}** achieved the highest $R^2$ score "
            f"({best_metrics.get('R2'):.4f}) and lowest prediction errors, effectively balancing bias and variance."
        )

# -------------------------------------------------------------
# TAB 2: DIAGNOSTICS (ACTUAL VS PREDICTED & RESIDUALS)
# -------------------------------------------------------------
with tab_diagnostics:
    st.subheader("Model Goodness-of-Fit Diagnostics")

    # Select which model to inspect
    available_models = list(metadata.get("test_predictions", {}).keys())
    selected_eval_model = st.selectbox("Select Model to Inspect:", available_models, index=0)

    actuals = metadata.get("test_actuals", [])
    predictions = metadata.get("test_predictions", {}).get(selected_eval_model, [])

    if actuals and predictions:
        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            st.write("##### Actual vs. Predicted Marks")
            fig_avp = plot_actual_vs_predicted(actuals, predictions, selected_eval_model)
            st.plotly_chart(fig_avp, use_container_width=True)
            st.caption("Points tightly clustered around the dashed red line indicate strong predictive alignment.")

        with diag_col2:
            st.write("##### Residual Analysis (Errors vs Predicted)")
            fig_res = plot_residuals(actuals, predictions, selected_eval_model)
            st.plotly_chart(fig_res, use_container_width=True)
            st.caption("A random, symmetrical scatter around zero without clear funnel patterns confirms homoscedasticity.")

        st.markdown("---")
        st.write("##### Residual Error Distribution")
        fig_dist = plot_residual_distribution(actuals, predictions, selected_eval_model)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption("A bell-shaped distribution centered at 0 validates normal distribution of error terms.")

# -------------------------------------------------------------
# TAB 3: FEATURE IMPORTANCE
# -------------------------------------------------------------
with tab_features:
    st.subheader("Feature Relative Importance & Impact")
    st.markdown(
        "Shows the relative importance / normalized weight of each feature in driving the model's final predictions."
    )

    feat_importances = metadata.get("feature_importances", {})
    if feat_importances:
        feat_df = pd.DataFrame([
            {"Feature": k, "Importance": v} for k, v in feat_importances.items()
        ]).sort_values(by="Importance", ascending=True)

        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Relative Feature Importance in Predicting Final Exam Marks",
            color="Importance",
            color_continuous_scale="Viridis",
            template="plotly_white",
        )
        fig_feat.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_feat, use_container_width=True)

        st.write("##### Detailed Importance Table")
        st.dataframe(feat_df.sort_values(by="Importance", ascending=False), use_container_width=True)
    else:
        st.info("Feature importance data will display after training.")

# -------------------------------------------------------------
# TAB 4: VIVA VOCE GLOSSARY
# -------------------------------------------------------------
with tab_viva:
    st.subheader("📚 Regression Metrics Glossary (For College Viva)")
    st.markdown("Use these precise definitions when explaining model evaluation to external examiners.")

    explanations = get_metric_explanations()
    for metric_key, info in explanations.items():
        with st.expander(f"📌 {info['name']} ({metric_key})", expanded=True):
            st.markdown(f"**Mathematical Formula:** `{info['formula']}`")
            st.markdown(f"**Description:** {info['explanation']}")
            st.info(f"**Viva Explanation:** {info['interpretation']}")
