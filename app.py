"""
Student Performance Prediction Using Machine Learning
Home Dashboard & Academic Analytics Portal
B.Tech Computer Science & Engineering Final-Year Project
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_data
from src.prediction import load_model_metadata, PERFORMANCE_TIERS

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction - Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Academic Analytics Theme
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #F8FAFC;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
    }
    /* Banner Header */
    .hero-banner {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 50%, #6366F1 100%);
        border-radius: 14px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 800;
        color: #FFFFFF;
    }
    .hero-banner p {
        margin: 8px 0 0 0;
        font-size: 1.05rem;
        color: #E0E7FF;
    }
    .sub-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 10px;
        color: #FFFFFF;
    }
    /* Section Headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1E293B;
        border-left: 4px solid #2563EB;
        padding-left: 10px;
        margin: 25px 0 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_dataset():
    """Load and cache dataset for rapid page navigation."""
    dataset_path = PROJECT_ROOT / "data" / "student_performance.csv"
    if not dataset_path.is_file():
        return None
    return load_data(dataset_path)


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=70)
        st.title("Academic Analytics")
        st.caption("B.Tech Final-Year Capstone Project")
        st.markdown("---")
        st.markdown(
            """
            **Project Navigation:**
            - 🏠 **Home Dashboard**
            - 📊 **1. Dataset Explorer**
            - 📈 **2. Model Evaluation**
            - 🎯 **3. Predict Performance**
            - 📜 **4. Prediction History**
            """
        )
        st.markdown("---")
        metadata = load_model_metadata()
        if metadata:
            st.success(f"**Active Model:**\n{metadata.get('selected_model', 'Trained Model')}")
            st.info(f"**Test R² Score:** {metadata.get('best_metrics', {}).get('R2', 0.0):.4f}")
        else:
            st.warning("Model metadata pending.")

        st.markdown("---")
        st.caption("Department of Computer Science & Engineering\nAcademic Year 2026")

    # Hero Banner
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Student Performance Prediction Using Machine Learning</h1>
            <p>An intelligent decision-support system analyzing academic indicators, study behaviors, and historical scores to forecast student examination performance.</p>
            <span class="sub-badge">B.Tech Computer Science & Engineering Final-Year Project</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = get_cached_dataset()

    if df is None:
        st.error("⚠️ Dataset not found at `data/student_performance.csv`. Please run `python data/generate_dataset.py`.")
        return

    # Categorize students for dashboard breakdown
    def assign_tier(marks):
        if marks >= 85.0:
            return "Excellent"
        elif marks >= 70.0:
            return "Good"
        elif marks >= 50.0:
            return "Average"
        return "Needs Improvement"

    df["Performance_Category"] = df["Final_Exam_Marks"].apply(assign_tier)

    # 4-Column Metric Cards
    total_students = len(df)
    avg_marks = df["Final_Exam_Marks"].mean()
    avg_attendance = df["Attendance_Rate"].mean()
    avg_study_hrs = df["Study_Hours_Per_Day"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Student Records</div>
                <div class="metric-val">{total_students:,}</div>
                <div style="color: #94A3B8; font-size: 0.8rem;">Cohort Sample Size</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Final Marks</div>
                <div class="metric-val">{avg_marks:.1f}</div>
                <div style="color: #94A3B8; font-size: 0.8rem;">Scale: 0 – 100 Marks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Attendance</div>
                <div class="metric-val">{avg_attendance:.1f}%</div>
                <div style="color: #94A3B8; font-size: 0.8rem;">Eligibility threshold: 75%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Study Hours</div>
                <div class="metric-val">{avg_study_hrs:.1f} <span style="font-size: 1.1rem;">hrs/day</span></div>
                <div style="color: #94A3B8; font-size: 0.8rem;">Self-Study Duration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

    # Visualizations Row
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.subheader("📊 Final Marks Distribution")
        fig_dist = px.histogram(
            df,
            x="Final_Exam_Marks",
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#3B82F6"],
            labels={"Final_Exam_Marks": "Final Examination Marks (out of 100)"},
        )
        fig_dist.update_layout(
            template="plotly_white",
            xaxis_title="Final Marks (0 - 100)",
            yaxis_title="Student Count",
            margin=dict(l=20, r=20, t=30, b=20),
            height=360,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with chart_col2:
        st.subheader("🎯 Performance Category Breakdown")
        tier_counts = df["Performance_Category"].value_counts().reset_index()
        tier_counts.columns = ["Category", "Count"]

        color_map = {
            "Excellent": "#10B981",
            "Good": "#3B82F6",
            "Average": "#F59E0B",
            "Needs Improvement": "#EF4444",
        }

        fig_pie = px.pie(
            tier_counts,
            values="Count",
            names="Category",
            color="Category",
            color_discrete_map=color_map,
            hole=0.45,
        )
        fig_pie.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Project Architecture & Workflow Overview
    st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
    st.subheader("🏛️ Project Architecture & Modules")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.info(
            """
            **1. Dataset Explorer**
            - Complete dataset preview
            - Missing value & duplicate audit
            - Correlation heatmaps
            - Attendance vs marks scatter
            """
        )

    with m_col2:
        st.info(
            """
            **2. Model Evaluation**
            - Linear Regression
            - Decision Tree Regressor
            - Random Forest Regressor
            - MAE, RMSE & R² comparisons
            """
        )

    with m_col3:
        st.info(
            """
            **3. Predict Performance**
            - Interactive student parameters
            - Instant score calculation
            - Actionable advice & feedback
            - Preset profiles for viva demo
            """
        )

    with m_col4:
        st.info(
            """
            **4. Prediction History**
            - SQLite database records
            - Date & category filtering
            - Export predictions to CSV
            - Record management & audit
            """
        )

    # Academic Advisory Note
    st.markdown("---")
    st.warning(
        "**Academic Advisory Disclaimer:** Predictions provided by this system are statistical estimates based on historical dataset patterns. "
        "They are designed as pedagogical decision-support tools for educators and students, and do not constitute a guarantee or formal academic outcome."
    )


if __name__ == "__main__":
    main()
