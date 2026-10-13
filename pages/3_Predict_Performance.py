"""
Module 6 & 7: Student Performance Prediction & Actionable Insights
------------------------------------------------------------------
Interactive input form with quick presets, pipeline inference,
performance tier categorization, and non-judgmental guidance.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db import save_prediction
from src.prediction import (
    PERFORMANCE_TIERS,
    categorize_performance,
    generate_student_insights,
    load_model_metadata,
    load_model_pipeline,
    predict_student_performance,
)

st.set_page_config(
    page_title="Predict Student Performance",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Student Performance Prediction & Academic Advisory")
st.caption("Input student academic behaviors and indicators to forecast examination outcomes with actionable recommendations.")

# Load Model Pipeline
try:
    pipeline = load_model_pipeline()
    metadata = load_model_metadata()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}. Please ensure model training was executed.")
    st.stop()

# Presets for Quick College Demo / Viva
PRESETS = {
    "Custom Input": {
        "attendance": 78.0,
        "study_hours": 3.5,
        "prev_marks": 68.0,
        "assignment": 72.0,
        "sleep": 7.0,
        "parent_edu": "Bachelor",
        "internet": "Yes",
        "extra": "Yes",
    },
    "🌟 High Achiever Student": {
        "attendance": 94.0,
        "study_hours": 6.5,
        "prev_marks": 88.0,
        "assignment": 92.0,
        "sleep": 7.5,
        "parent_edu": "Master",
        "internet": "Yes",
        "extra": "Yes",
    },
    "⚖️ Average Consistent Student": {
        "attendance": 80.0,
        "study_hours": 3.5,
        "prev_marks": 68.0,
        "assignment": 74.0,
        "sleep": 7.0,
        "parent_edu": "Bachelor",
        "internet": "Yes",
        "extra": "No",
    },
    "⚠️ At-Risk / Attendance Shortage Student": {
        "attendance": 55.0,
        "study_hours": 1.2,
        "prev_marks": 46.0,
        "assignment": 50.0,
        "sleep": 5.0,
        "parent_edu": "High School",
        "internet": "Yes",
        "extra": "No",
    },
}

with st.sidebar:
    st.markdown("### ⚡ Quick Demo Presets")
    st.caption("Select a simulated student profile for viva demonstration:")
    selected_preset_key = st.selectbox(
        "Load Preset Profile:",
        list(PRESETS.keys()),
        index=0,
    )
    preset = PRESETS[selected_preset_key]

    st.markdown("---")
    st.markdown("### ℹ️ Active Pipeline")
    st.write(f"**Model:** {metadata.get('selected_model', 'Linear Regression')}")
    st.write(f"**Trained Accuracy (R²):** {metadata.get('best_metrics', {}).get('R2', 0.0):.4f}")
    st.write(f"**Mean Error (MAE):** ±{metadata.get('best_metrics', {}).get('MAE', 0.0):.2f} Marks")

# Input Form
with st.form("student_prediction_form"):
    st.subheader("📝 Student Academic & Behavioral Parameters")

    col_tag, _ = st.columns([1, 2])
    with col_tag:
        student_tag = st.text_input(
            "Student Tag / Identifier (Optional):",
            value="Candidate-101",
            help="Anonymous reference tag (no private student names or personal data required).",
        )

    st.markdown("##### Numerical Academic Attributes")
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        attendance = st.slider(
            "Attendance Rate (%):",
            min_value=40.0,
            max_value=100.0,
            value=float(preset["attendance"]),
            step=0.5,
            help="Percentage of lectures attended in the semester.",
        )
        study_hours = st.slider(
            "Daily Study Hours (hrs/day):",
            min_value=0.5,
            max_value=12.0,
            value=float(preset["study_hours"]),
            step=0.5,
            help="Average self-study duration per day.",
        )

    with f_col2:
        prev_marks = st.slider(
            "Previous Exam Marks (0 - 100):",
            min_value=30.0,
            max_value=100.0,
            value=float(preset["prev_marks"]),
            step=0.5,
            help="Score achieved in the previous semester or midterm examination.",
        )
        assignment_score = st.slider(
            "Assignment / Internal Score (0 - 100):",
            min_value=30.0,
            max_value=100.0,
            value=float(preset["assignment"]),
            step=0.5,
            help="Cumulative coursework and lab assignment marks.",
        )

    with f_col3:
        sleep_hours = st.slider(
            "Average Sleep Duration (hrs/day):",
            min_value=4.0,
            max_value=12.0,
            value=float(preset["sleep"]),
            step=0.5,
            help="Daily average sleep duration.",
        )

    st.markdown("##### Environmental & Background Attributes")
    c_col1, c_col2, c_col3 = st.columns(3)

    with c_col1:
        parent_edu = st.selectbox(
            "Parent Education Level:",
            ["High School", "Bachelor", "Master", "Doctorate"],
            index=["High School", "Bachelor", "Master", "Doctorate"].index(preset["parent_edu"]),
        )

    with c_col2:
        internet = st.selectbox(
            "Home Internet Access:",
            ["Yes", "No"],
            index=["Yes", "No"].index(preset["internet"]),
        )

    with c_col3:
        extracurricular = st.selectbox(
            "Extracurricular Activities Participation:",
            ["Yes", "No"],
            index=["Yes", "No"].index(preset["extra"]),
        )

    submitted = st.form_submit_button("🔮 Predict Student Performance", use_container_width=True, type="primary")

# Execute Prediction
if submitted or "last_prediction" in st.session_state:
    if submitted:
        input_data = {
            "Attendance_Rate": attendance,
            "Study_Hours_Per_Day": study_hours,
            "Previous_Exam_Marks": prev_marks,
            "Assignment_Score": assignment_score,
            "Sleep_Hours_Per_Day": sleep_hours,
            "Parent_Education_Level": parent_edu,
            "Internet_Access": internet,
            "Extracurricular_Activities": extracurricular,
        }

        try:
            prediction_result = predict_student_performance(input_data, pipeline=pipeline)
            prediction_result["student_tag"] = student_tag
            st.session_state["last_prediction"] = prediction_result
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            st.stop()

    result = st.session_state["last_prediction"]
    pred_marks = result["predicted_marks"]
    tier = result["performance_tier"]
    tier_color = result["tier_color"]
    badge_type = result["tier_badge"]
    features_used = result["input_features"]
    active_tag = result.get("student_tag", "Anonymous")

    st.markdown("---")
    st.subheader("🎓 Prediction Result & Performance Assessment")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        # Visual Gauge using Plotly
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pred_marks,
                title={"text": "Predicted Final Marks", "font": {"size": 22}},
                delta={"reference": features_used["Previous_Exam_Marks"], "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1E293B"},
                    "bar": {"color": tier_color},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "#E2E8F0",
                    "steps": [
                        {"range": [0, 50], "color": "#FEE2E2"},
                        {"range": [50, 70], "color": "#FEF3C7"},
                        {"range": [70, 85], "color": "#DBEAFE"},
                        {"range": [85, 100], "color": "#D1FAE5"},
                    ],
                    "threshold": {
                        "line": {"color": "#1E293B", "width": 4},
                        "thickness": 0.75,
                        "value": pred_marks,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 2px solid {tier_color}; border-radius: 12px; padding: 22px; margin-top: 15px;">
                <h4 style="margin: 0; color: #1E293B;">Performance Classification:</h4>
                <div style="font-size: 2.2rem; font-weight: 800; color: {tier_color}; margin: 8px 0;">{tier}</div>
                <p style="margin: 0 0 10px 0; color: #475569;">
                    <b>Predicted Score:</b> {pred_marks:.1f} / 100 Marks<br>
                    <b>Expected Accuracy Range:</b> {pred_marks - metadata.get('best_metrics', {}).get('MAE', 2.3):.1f} to {pred_marks + metadata.get('best_metrics', {}).get('MAE', 2.3):.1f} Marks (±MAE)
                </p>
                <div style="font-size: 0.85rem; color: #64748B;">
                    Threshold Scale: Excellent (85-100) | Good (70-84) | Average (50-69) | Needs Improvement (&lt;50)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Database Save and Export Buttons
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("💾 Save Record to Database", use_container_width=True):
                rec_id = save_prediction(
                    features=features_used,
                    predicted_marks=pred_marks,
                    performance_category=tier,
                    student_tag=active_tag,
                    model_used=metadata.get("selected_model", "Linear Regression Pipeline"),
                )
                st.toast(f"Record saved with ID #{rec_id} in SQLite history!", icon="✅")

        with b_col2:
            summary_txt = (
                f"STUDENT PERFORMANCE PREDICTION REPORT\n"
                f"Student Tag: {active_tag}\n"
                f"Predicted Marks: {pred_marks} / 100\n"
                f"Performance Category: {tier}\n"
                f"Model: {metadata.get('selected_model')}\n\n"
                f"INPUT PARAMETERS:\n"
                f"- Attendance Rate: {features_used['Attendance_Rate']}%\n"
                f"- Daily Study Hours: {features_used['Study_Hours_Per_Day']} hrs\n"
                f"- Previous Exam Marks: {features_used['Previous_Exam_Marks']}\n"
                f"- Assignment Score: {features_used['Assignment_Score']}\n"
                f"- Sleep Hours: {features_used['Sleep_Hours_Per_Day']} hrs\n"
                f"- Parental Education: {features_used['Parent_Education_Level']}\n"
                f"- Internet Access: {features_used['Internet_Access']}\n"
                f"- Extracurricular: {features_used['Extracurricular_Activities']}\n\n"
                f"DISCLAIMER: Predictions are statistical estimates for educational decision support."
            )
            st.download_button(
                label="📄 Download Prediction Summary",
                data=summary_txt,
                file_name=f"prediction_{active_tag}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # Module 7: Actionable Student Insights
    st.markdown("---")
    st.subheader("💡 Actionable Academic Insights & Guidance")
    st.caption("Personalized pedagogical feedback based on behavioral indicators:")

    insights_list = result["insights"]
    ins_cols = st.columns(len(insights_list) if len(insights_list) <= 3 else 3)

    for i, ins in enumerate(insights_list):
        col_idx = i % 3
        with ins_cols[col_idx]:
            if ins["type"] == "warning":
                st.warning(f"**{ins['title']}**\n\n{ins['detail']}")
            elif ins["type"] == "success":
                st.success(f"**{ins['title']}**\n\n{ins['detail']}")
            else:
                st.info(f"**{ins['title']}**\n\n{ins['detail']}")

    # Academic Advisory Note
    st.markdown("---")
    st.caption(
        "⚠️ **Ethical AI Disclaimer:** Statistical predictions evaluate historical trends and do not define an individual's potential. "
        "These metrics should be used solely to identify areas of academic support and constructive encouragement."
    )
