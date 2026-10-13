"""
Module 2 & 3: Dataset Explorer & Exploratory Data Analysis (EDA)
---------------------------------------------------------------
Provides in-depth exploratory data analysis, dataset profiling,
correlation analysis, and student cohort behavior visualizations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    load_data,
)

st.set_page_config(
    page_title="Dataset Explorer & EDA",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dataset Management & Exploratory Data Analysis")
st.caption("Inspect raw cohort data, schema properties, statistical distributions, and pedagogical correlations.")

DATASET_PATH = PROJECT_ROOT / "data" / "student_performance.csv"

if not DATASET_PATH.is_file():
    st.error(f"❌ Dataset file not found at: `{DATASET_PATH.resolve()}`. Run `python data/generate_dataset.py` to generate it.")
    st.stop()

# Load data
df = load_data(DATASET_PATH)

# Categorize performance
def assign_tier(marks):
    if marks >= 85.0:
        return "Excellent"
    elif marks >= 70.0:
        return "Good"
    elif marks >= 50.0:
        return "Average"
    return "Needs Improvement"

df["Performance_Category"] = df["Final_Exam_Marks"].apply(assign_tier)

# Attendance Groups
df["Attendance_Group"] = pd.cut(
    df["Attendance_Rate"],
    bins=[0, 74.99, 85.0, 100],
    labels=["Low Attendance (<75%)", "Moderate Attendance (75-85%)", "High Attendance (>85%)"],
)

# Tabs for structured exploration
tab_overview, tab_eda, tab_correlations, tab_cohorts = st.tabs([
    "📋 Dataset Profiling & Summary",
    "📈 Feature vs Marks (Scatter & Trends)",
    "🔥 Correlation Heatmap",
    "👥 Cohort & Group Comparisons",
])

# -------------------------------------------------------------
# TAB 1: DATASET PROFILING
# -------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset Ingestion & Data Health")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records (Rows)", f"{len(df):,}")
    with c2:
        st.metric("Total Features (Columns)", f"{len(df.columns)}")
    with c3:
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_count}")
    with c4:
        dup_count = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{dup_count}")

    st.markdown("---")

    # Preview
    st.write("##### Sample Records Preview")
    rows_to_show = st.slider("Number of preview rows:", min_value=5, max_value=50, value=10, step=5)
    st.dataframe(df.head(rows_to_show), use_container_width=True)

    # Download Dataset Button
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Complete Dataset (CSV)",
        data=csv_bytes,
        file_name="student_performance_dataset.csv",
        mime="text/csv",
        help="Download the entire cleaned dataset for offline analysis in Excel or Python.",
    )

    st.markdown("---")
    st.write("##### Descriptive Statistics (Numerical Attributes)")
    st.dataframe(df[NUMERICAL_FEATURES + [TARGET_COLUMN]].describe().round(2), use_container_width=True)

    st.write("##### Schema & Feature Data Types")
    schema_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.notnull().sum(),
        "Unique Values": df.nunique(),
    })
    st.dataframe(schema_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
with tab_eda:
    st.subheader("Key Academic Drivers vs. Final Exam Marks")
    st.markdown(
        "Explore how daily study routines, lecture attendance, previous test results, "
        "and coursework impact final marks."
    )

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        # Attendance vs Marks
        fig_att = px.scatter(
            df,
            x="Attendance_Rate",
            y="Final_Exam_Marks",
            color="Performance_Category",
            trendline="ols",
            title="Attendance Rate (%) vs. Final Marks",
            labels={
                "Attendance_Rate": "Attendance Rate (%)",
                "Final_Exam_Marks": "Final Marks (0 - 100)",
            },
            color_discrete_map={
                "Excellent": "#10B981",
                "Good": "#3B82F6",
                "Average": "#F59E0B",
                "Needs Improvement": "#EF4444",
            },
            template="plotly_white",
        )
        fig_att.add_vline(x=75, line_dash="dash", line_color="#DC2626", annotation_text="75% Min. Eligibility")
        st.plotly_chart(fig_att, use_container_width=True)
        st.caption("💡 **Interpretation:** Strong positive correlation between class attendance and final marks. Students below the 75% cutoff disproportionately fall in the 'Needs Improvement' zone.")

    with eda_col2:
        # Study Hours vs Marks
        fig_study = px.scatter(
            df,
            x="Study_Hours_Per_Day",
            y="Final_Exam_Marks",
            color="Performance_Category",
            trendline="ols",
            title="Daily Study Hours vs. Final Marks",
            labels={
                "Study_Hours_Per_Day": "Daily Study Hours (hrs/day)",
                "Final_Exam_Marks": "Final Marks (0 - 100)",
            },
            color_discrete_map={
                "Excellent": "#10B981",
                "Good": "#3B82F6",
                "Average": "#F59E0B",
                "Needs Improvement": "#EF4444",
            },
            template="plotly_white",
        )
        st.plotly_chart(fig_study, use_container_width=True)
        st.caption("💡 **Interpretation:** Daily study hours exhibit a solid upward trend. Marginal performance gains plateau slightly beyond 7 hours/day due to fatigue.")

    eda_col3, eda_col4 = st.columns(2)

    with eda_col3:
        # Previous Marks vs Final Marks
        fig_prev = px.scatter(
            df,
            x="Previous_Exam_Marks",
            y="Final_Exam_Marks",
            color="Performance_Category",
            trendline="ols",
            title="Previous Exam Marks vs. Final Marks",
            labels={
                "Previous_Exam_Marks": "Previous Exam Marks (0 - 100)",
                "Final_Exam_Marks": "Final Marks (0 - 100)",
            },
            color_discrete_map={
                "Excellent": "#10B981",
                "Good": "#3B82F6",
                "Average": "#F59E0B",
                "Needs Improvement": "#EF4444",
            },
            template="plotly_white",
        )
        st.plotly_chart(fig_prev, use_container_width=True)
        st.caption("💡 **Interpretation:** Historical performance is one of the highest individual predictors of future exam outcomes.")

    with eda_col4:
        # Assignment Score vs Final Marks
        fig_assign = px.scatter(
            df,
            x="Assignment_Score",
            y="Final_Exam_Marks",
            color="Performance_Category",
            trendline="ols",
            title="Assignment Scores vs. Final Marks",
            labels={
                "Assignment_Score": "Assignment Score (0 - 100)",
                "Final_Exam_Marks": "Final Marks (0 - 100)",
            },
            color_discrete_map={
                "Excellent": "#10B981",
                "Good": "#3B82F6",
                "Average": "#F59E0B",
                "Needs Improvement": "#EF4444",
            },
            template="plotly_white",
        )
        st.plotly_chart(fig_assign, use_container_width=True)
        st.caption("💡 **Interpretation:** Continuous evaluation marks strongly align with final performance, reinforcing the importance of timely homework submission.")

# -------------------------------------------------------------
# TAB 3: CORRELATION HEATMAP
# -------------------------------------------------------------
with tab_correlations:
    st.subheader("Numerical Features Correlation Matrix")
    st.markdown(
        "Pearson correlation coefficients ($r$) between numerical features and the target variable. "
        "Values close to $+1.0$ indicate strong positive association."
    )

    corr_df = df[NUMERICAL_FEATURES + [TARGET_COLUMN]].corr().round(3)

    fig_heatmap = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Heatmap",
    )
    fig_heatmap.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.write("##### Correlation with Final Exam Marks (Ranked)")
    target_corrs = corr_df[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
    target_corrs_df = pd.DataFrame({
        "Feature": target_corrs.index,
        "Correlation with Final Marks (r)": target_corrs.values,
        "Relationship Strength": [
            "Strong Positive" if abs(v) > 0.6 else "Moderate Positive" if abs(v) > 0.3 else "Weak"
            for v in target_corrs.values
        ],
    })
    st.dataframe(target_corrs_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: COHORT & GROUP COMPARISONS
# -------------------------------------------------------------
with tab_cohorts:
    st.subheader("Categorical Cohort Insights")

    grp_col1, grp_col2 = st.columns(2)

    with grp_col1:
        st.write("##### Attendance Tiers vs. Final Marks Distribution")
        fig_box = px.box(
            df,
            x="Attendance_Group",
            y="Final_Exam_Marks",
            color="Attendance_Group",
            title="Marks Distribution Across Attendance Thresholds",
            color_discrete_sequence=["#EF4444", "#F59E0B", "#10B981"],
            template="plotly_white",
        )
        fig_box.update_layout(showlegend=False, xaxis_title="Attendance Group", yaxis_title="Final Marks")
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("Students with >85% attendance achieve a substantially higher median exam score.")

    with grp_col2:
        st.write("##### Impact of Sleep Hours on Final Marks")
        # Round sleep hours to nearest whole integer for grouping
        df["Sleep_Binned"] = df["Sleep_Hours_Per_Day"].round().astype(int)
        sleep_agg = df.groupby("Sleep_Binned")["Final_Exam_Marks"].agg(["mean", "count"]).reset_index()
        fig_sleep = px.bar(
            sleep_agg,
            x="Sleep_Binned",
            y="mean",
            text=sleep_agg["mean"].round(1),
            title="Average Final Marks by Daily Sleep Duration",
            labels={"Sleep_Binned": "Sleep Hours / Night (Approx)", "mean": "Average Final Marks"},
            color="mean",
            color_continuous_scale="Blues",
            template="plotly_white",
        )
        fig_sleep.update_layout(yaxis_range=[40, 75])
        st.plotly_chart(fig_sleep, use_container_width=True)
        st.caption("Optimal performance is observed around 7 to 8 hours of sleep. Severe sleep deprivation (< 5 hrs) shows marked decline.")
