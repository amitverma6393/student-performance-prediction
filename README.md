# 🎓 Student Performance Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-14%20Passed-brightgreen.svg)](tests/)

> **A complete, production-grade Machine Learning capstone project for B.Tech Computer Science & Engineering final-year college submission, demonstration, and viva voce defense.**

---

## 📌 Project Overview

**Student Performance Prediction Using Machine Learning** is an interactive web portal designed to forecast a student's semester examination performance on a **0–100 continuous marks scale** using behavioral, academic, and environmental factors:
- **Class Attendance Rate (%)**
- **Daily Self-Study Hours (hrs/day)**
- **Previous Examination Marks (0–100)**
- **Internal Assignment & Coursework Scores (0–100)**
- **Daily Sleep Duration (hrs/day)**
- **Parental Educational Qualification**
- **Home Internet Connectivity**
- **Extracurricular Activities Participation**

The system trains and benchmarks three candidate models (**Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor**) using Scikit-learn Pipelines, identifies the optimal model, and deploys it within a multi-page **Streamlit** dashboard. Predictions are mapped to academic grading tiers (Excellent, Good, Average, Needs Improvement) accompanied by actionable, non-judgmental guidance and a local **SQLite** history audit log.

---

## 🏗️ Project Folder Structure

```text
student_performance_prediction/
├── app.py                                   # Home Dashboard & Academic Overview
├── requirements.txt                         # Pinned Python package dependencies
├── README.md                                # Complete project documentation & guide
├── .gitignore                               # Git ignore configuration
├── data/
│   ├── generate_dataset.py                  # Deterministic dataset generation script
│   └── student_performance.csv              # Documented dataset (1,500 student records)
├── models/
│   ├── student_performance_pipeline.joblib  # Serialized best Scikit-learn Pipeline
│   ├── candidate_models.joblib              # Serialized candidate model pipelines
│   └── model_metadata.json                  # Training metrics, schema, and diagnostics
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py                # Schema validation, ColumnTransformer & splits
│   ├── train_model.py                       # Training, benchmarking, and export pipeline
│   ├── evaluation.py                        # Metric calculations (MAE, RMSE, R²) & Plotly plots
│   └── prediction.py                        # Inference engine, categorization & insights
├── pages/
│   ├── 1_Dataset_Explorer.py                # Dataset inspection, stats, and interactive EDA
│   ├── 2_Model_Evaluation.py                # Cross-model comparisons & residual diagnostics
│   ├── 3_Predict_Performance.py             # Prediction form, presets, gauge & feedback
│   └── 4_Prediction_History.py              # SQLite audit trail, filters, and CSV export
├── database/
│   ├── __init__.py
│   ├── db.py                                # SQLite CRUD operations and connection pooling
│   └── predictions.db                       # Auto-generated runtime SQLite database
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py                # Unit tests for preprocessing & validation
│   └── test_prediction.py                   # Unit tests for inference, tiers & database
└── docs/
    ├── project_report.md                    # 22-section B.Tech CSE formal project report
    ├── system_architecture.md               # Mermaid architecture diagrams & DFDs
    └── viva_questions.md                    # 30+ Viva questions & seminar presentation outline
```

---

## ⚡ Tech Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10+ / 3.11+ / 3.12+ / 3.13+ | Core programming runtime |
| **Data Processing** | Pandas, NumPy | Latest stable | Data manipulation and numerical arrays |
| **Machine Learning** | Scikit-learn | 1.4+ | Preprocessing pipelines, scaling & regressors |
| **Model Persistence** | Joblib | 1.3+ | Serialization of trained pipelines |
| **Visualizations** | Plotly, Matplotlib, Seaborn | Latest stable | Interactive diagnostic charts & heatmaps |
| **Web Interface** | Streamlit | 1.35+ | Multi-page modern analytics UI |
| **Database** | SQLite3 | Built-in | Local prediction audit logs & history |
| **Testing** | Pytest | 8.0+ | Automated unit and integration testing |

---

## 📊 Model Evaluation & Benchmarks

All models were evaluated on an identical held-out test dataset (300 records) using an 80/20 train-test split (`random_state=42`):

| Model Algorithm | $R^2$ Score (Higher is better) | MAE (Marks) | RMSE (Marks) | Status |
|---|---|---|---|---|
| **Linear Regression** | **0.9001** | **2.3309** | **2.9046** | 🏆 **Best Model (Selected)** |
| **Random Forest Regressor** | **0.8530** | **2.7516** | **3.5239** | Alternative Ensemble |
| **Decision Tree Regressor** | **0.7852** | **3.2612** | **4.2595** | Baseline Non-Parametric |

### Grading Tier Classification

Predictions are classified into standard academic tiers:
- **Excellent:** $85.0 - 100.0$ Marks (Green)
- **Good:** $70.0 - 84.9$ Marks (Blue)
- **Average:** $50.0 - 69.9$ Marks (Amber)
- **Needs Improvement:** Below $50.0$ Marks (Red)

---

## 🚀 Quick Start Guide (Windows & VS Code)

### Step 1: Open Project in VS Code
Open your terminal in VS Code (`Ctrl + ~`) inside the project folder:
```powershell
cd "c:\Users\verma\OneDrive\Desktop\student_performance_prediction"
```

### Step 2: (Optional) Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Generate the Dataset (If not already present)
```powershell
python data/generate_dataset.py
```

### Step 5: Train and Persist the Machine Learning Pipeline
```powershell
python src/train_model.py
```

### Step 6: Run Automated Tests
```powershell
pytest -v
```
*(All 14 unit tests will pass successfully!)*

### Step 7: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 🖥️ Multi-Page Streamlit Features

1. **🏠 Home Dashboard (`app.py`):**
   - KPI metric cards (Total Records, Average Marks, Average Attendance, Daily Study Hours).
   - Marks distribution histogram and performance tier donut chart.
   - Quick architectural overview.
2. **📊 1. Dataset Explorer (`pages/1_Dataset_Explorer.py`):**
   - Full dataset preview with configurable row slider.
   - Missing value and duplicate audits.
   - Interactive scatter plots with trendlines (Attendance vs Marks, Study vs Marks, Previous Marks vs Final Marks).
   - Pearson correlation heatmap and CSV dataset download button.
3. **📈 2. Model Evaluation (`pages/2_Model_Evaluation.py`):**
   - Candidate models comparison table (Linear Regression, Decision Tree, Random Forest).
   - Actual vs. Predicted scatter plot with $y = x$ reference diagonal.
   - Residual error distribution histogram.
   - Relative feature importance chart.
   - Viva Voce metric glossary explaining $R^2$, MAE, and RMSE.
4. **🎯 3. Predict Performance (`pages/3_Predict_Performance.py`):**
   - Interactive parameter input sliders and selectors.
   - **Quick Demo Presets for Viva:** "High Achiever", "Average Student", "At-Risk Student".
   - Visual score gauge and tier badge.
   - Actionable, non-judgmental guidance cards.
   - "Save Record to Database" and "Download Summary" buttons.
5. **📜 4. Prediction History (`pages/4_Prediction_History.py`):**
   - Historical inference audit records from SQLite.
   - Category filtering and score distribution plots.
   - Download history as CSV.
   - Single-record deletion and clear database controls.

---

## ☁️ Deployment Instructions (Streamlit Community Cloud)

1. **Initialize Git & Push to GitHub:**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit of Student Performance Prediction project"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/student-performance-prediction.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud:**
   - Navigate to [share.streamlit.io](https://share.streamlit.io/).
   - Click **"New App"**.
   - Select your GitHub repository and branch (`main`).
   - Set Main file path to: `app.py`.
   - Click **"Deploy!"**.
   - The platform will install dependencies from `requirements.txt` and launch your cloud web app!

---

## 🎓 College Viva & Submission Deliverables

Complete university documentation is provided inside the `docs/` directory:
- 📄 **`docs/project_report.md`**: Complete 22-section B.Tech CSE formal project report (Title page, Certificate, DFD, Use Case, Math equations, actual metrics, references).
- 🏛️ **`docs/system_architecture.md`**: Mermaid architecture diagrams, 3-tier layout, pipeline flow, and SQLite ER diagram.
- 🗣️ **`docs/viva_questions.md`**: 30+ Viva Voce questions with detailed answers + 10-slide seminar presentation outline.

---

## ⚖️ Ethical AI & Academic Disclaimer

> **Academic Advisory Note:** Predictions produced by this application are statistical estimations based on historical cohort correlations. They are designed to function as an educational decision-support tool to encourage proactive study habits and identify opportunities for remedial assistance. Predictions do not constitute deterministic or legally binding academic grades.
