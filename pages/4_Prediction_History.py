"""
Module 8 & 9: Prediction History & SQLite Record Management
-----------------------------------------------------------
View, filter, export, and manage historical student performance predictions
persisted in the local SQLite database.
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db import (
    clear_all_predictions,
    delete_prediction,
    get_all_predictions,
    init_db,
)

st.set_page_config(
    page_title="Prediction History & Audit Log",
    page_icon="📜",
    layout="wide",
)

st.title("📜 Prediction History & Audit Records")
st.caption("Manage saved student predictions, filter by category, export CSV audit logs, or prune records.")

# Ensure DB is initialized
init_db()

# Filter Controls
with st.sidebar:
    st.markdown("### 🔍 Filter Records")
    category_filter = st.selectbox(
        "Filter by Performance Category:",
        ["All", "Excellent", "Good", "Average", "Needs Improvement"],
        index=0,
    )
    records_limit = st.slider("Max Records to Display:", min_value=10, max_value=500, value=100, step=10)

    st.markdown("---")
    st.markdown("### ⚠️ Database Maintenance")
    if st.button("🗑️ Clear Entire History", help="Delete all records permanently"):
        clear_all_predictions()
        st.toast("Database records cleared successfully!", icon="🗑️")
        st.rerun()

# Fetch predictions
df_history = get_all_predictions(category_filter=category_filter, limit=records_limit)

if df_history.empty:
    st.info(
        "ℹ️ No prediction records found in the history database yet.\n\n"
        "Go to **Page 3: Predict Performance**, run a prediction, and click **'Save Record to Database'** to log entries here!"
    )
else:
    # Top Stats
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    with h_col1:
        st.metric("Total Logged Records", f"{len(df_history)}")
    with h_col2:
        st.metric("Average Predicted Marks", f"{df_history['predicted_marks'].mean():.1f}")
    with h_col3:
        st.metric("Highest Prediction", f"{df_history['predicted_marks'].max():.1f}")
    with h_col4:
        st.metric("Lowest Prediction", f"{df_history['predicted_marks'].min():.1f}")

    st.markdown("---")

    # Display History Table
    st.write("##### Historical Inference Records")
    display_cols = [
        "id",
        "timestamp",
        "student_tag",
        "predicted_marks",
        "performance_category",
        "attendance_rate",
        "study_hours",
        "prev_marks",
        "assignment_score",
        "model_used",
    ]
    st.dataframe(
        df_history[display_cols].rename(
            columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "student_tag": "Student Tag",
                "predicted_marks": "Predicted Marks",
                "performance_category": "Category",
                "attendance_rate": "Attendance (%)",
                "study_hours": "Study (hrs/d)",
                "prev_marks": "Prev Marks",
                "assignment_score": "Assignment Score",
                "model_used": "Model",
            }
        ),
        use_container_width=True,
    )

    # Export & Management Section
    col_export, col_delete = st.columns([1, 1])

    with col_export:
        st.write("##### 📥 Export History")
        csv_data = df_history.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction History as CSV",
            data=csv_data,
            file_name="student_predictions_audit_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_delete:
        st.write("##### 🗑️ Delete Specific Record")
        record_ids = df_history["id"].tolist()
        selected_id = st.selectbox("Select Record ID to Delete:", record_ids)
        if st.button("Delete Selected Record", type="secondary", use_container_width=True):
            success = delete_prediction(selected_id)
            if success:
                st.toast(f"Record #{selected_id} deleted!", icon="✅")
                st.rerun()
            else:
                st.error("Failed to delete record.")

    # Distribution Chart of Logged Predictions
    if len(df_history) >= 3:
        st.markdown("---")
        st.write("##### 📊 Distribution of Logged Predictions")
        fig_hist = px.histogram(
            df_history,
            x="predicted_marks",
            color="performance_category",
            nbins=20,
            title="Logged Prediction Score Distribution",
            color_discrete_map={
                "Excellent": "#10B981",
                "Good": "#3B82F6",
                "Average": "#F59E0B",
                "Needs Improvement": "#EF4444",
            },
            template="plotly_white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
