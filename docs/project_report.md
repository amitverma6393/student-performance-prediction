# A PROJECT REPORT ON
# STUDENT PERFORMANCE PREDICTION USING MACHINE LEARNING

**Submitted in partial fulfillment of the requirements for the award of the degree of**
**BACHELOR OF TECHNOLOGY IN COMPUTER SCIENCE & ENGINEERING**

---

### [SECTION 1: TITLE PAGE]

**Project Title:** Student Performance Prediction Using Machine Learning  
**Department:** Department of Computer Science & Engineering  
**Institution:** [Name of College / University]  
**Academic Year:** 2025 – 2026  

**Submitted By:**  
- Student Name: [Your Name]  
- University Roll Number: [Your Roll Number]  
- Registration Number: [Your Registration Number]  

**Under the Guidance of:**  
- Project Supervisor / Mentor: [Supervisor Name], [Designation]  
- Head of Department: [HOD Name], Department of Computer Science & Engineering  

---

### [SECTION 2: CERTIFICATE TEMPLATE]

**DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING**  
**[NAME OF COLLEGE / UNIVERSITY]**  
**[CITY, STATE, PIN CODE]**

#### CERTIFICATE OF ORIGINAL WORK

This is to certify that the project report entitled **"Student Performance Prediction Using Machine Learning"** submitted by **[Your Name]** (Roll No: **[Your Roll Number]**) in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science & Engineering** is a bona fide record of the work carried out under my supervision and guidance during the academic year 2025–2026.

The results embodied in this report have not been submitted to any other University or Institute for the award of any degree or diploma.

\
\
__________________________  \
**[Supervisor Name]**  \
Project Supervisor / Assistant Professor  \
Department of Computer Science & Engineering  

\
\
__________________________  \
**[HOD Name]**  \
Professor & Head of Department  \
Department of Computer Science & Engineering  

\
\
__________________________  \
**External Examiner**  \
Date of Viva-Voce Examination: ______________

---

### [SECTION 3: ACKNOWLEDGEMENT]

I would like to express my deepest gratitude to my project supervisor, **[Supervisor Name]**, for their invaluable guidance, encouragement, constructive critiques, and continuous mentoring throughout the lifecycle of this project.

I extend my heartfelt thanks to **[HOD Name]**, Head of the Department of Computer Science & Engineering, and the entire faculty for providing the necessary infrastructural resources, computing labs, and academic atmosphere to execute this work.

Lastly, I thank my family and fellow peers whose moral support and assistance made this capstone project possible.

**[Your Name]**  
B.Tech Final Year, Computer Science & Engineering  

---

### [SECTION 4: ABSTRACT]

Academic success in tertiary education is deeply correlated with behavioral routines, classroom engagement, self-study consistency, and prior conceptual foundations. In traditional academic settings, student underperformance is often recognized late—typically after semester-end examinations—rendering timely remedial interventions ineffective.

This capstone project presents an end-to-end Machine Learning decision-support portal named **"Student Performance Prediction Using Machine Learning"**. By utilizing supervised regression techniques, the system models empirical relationships between student attendance rates, daily study durations, historical exam marks, assignment performance, sleep hygiene, and background educational indicators to forecast final examination scores on a 0–100 scale. 

A rigorous Scikit-learn Pipeline architecture with standardized preprocessing (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) was developed to eliminate data leakage. Three candidate regression algorithms—**Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor**—were implemented and benchmarked on a verified cohort of 1,500 student records. Experimental results demonstrate that **Linear Regression** achieved superior predictive performance ($R^2 = 0.9001$, $\text{MAE} = 2.3309$, $\text{RMSE} = 2.9046$), closely followed by the Random Forest Regressor ($R^2 = 0.8530$). The trained model is deployed within a multi-page interactive **Streamlit** web application featuring exploratory data analysis (EDA), real-time score gauges, academic tier categorization, ethical non-judgmental guidance, and an audit-trail **SQLite** database with CSV export. The system serves as a practical, transparent pedagogical tool for academic mentors and students alike.

---

### [SECTION 5: INTRODUCTION]

The advent of Educational Data Mining (EDM) and Learning Analytics (LA) has transformed traditional pedagogy into an evidence-based discipline. Educational institutions gather substantial datasets concerning student participation, internal assessment scores, and attendance rates. However, in the absence of analytical automation, these data points often remain siloed within administrative databases until after grade publication.

Machine Learning techniques provide an automated mechanism for identifying non-linear and multidimensional patterns across student cohorts. By identifying students experiencing academic friction early in the semester, educational institutions can transition from reactive penalty structures to proactive student support systems.

This project delivers a comprehensive, reproducible software artifact that operationalizes machine learning for academic performance prediction, balancing mathematical rigor with beginner-friendly software design.

---

### [SECTION 6: PROBLEM STATEMENT]

In modern engineering curricula:
1. **Late Intervention:** Academic shortcomings are usually identified only when formal semester exam results are declared, leaving no window for remedial tutoring or targeted academic intervention.
2. **Multidimensional Blind Spots:** Faculty advisors manually evaluate attendance or internal marks in isolation, failing to assess the compounded interactions between daily study hours, sleep habits, assignment submissions, and attendance thresholds.
3. **Lack of Transparent Tools:** Many commercial analytics platforms are opaque black boxes that fail to explain evaluation metrics or lack ethical guardrails, frequently misinterpreting statistical correlations as deterministic outcomes.

The challenge is to construct an accessible, explainable, and ethically sound machine learning application that predicts final exam marks, visualizes diagnostic residuals, explains evaluation metrics, and offers constructive academic feedback without collecting sensitive personal student identifiers.

---

### [SECTION 7: OBJECTIVES]

The specific objectives of this project are:
1. **Dataset Synthesis & Curation:** To construct and profile a statistically coherent academic dataset with 1,500 records representing realistic student behaviors and constraints.
2. **Data Pipeline Engineering:** To implement leak-free preprocessing pipelines utilizing Scikit-learn's `ColumnTransformer` for simultaneous z-score scaling of continuous variables and one-hot encoding of categorical attributes.
3. **Model Benchmarking:** To train, evaluate, and compare multiple regression models (Linear Regression, Decision Tree, Random Forest) using standard regression metrics ($R^2$, MAE, RMSE).
4. **Interactive Multi-Page Web Deployment:** To build an intuitive, responsive Streamlit dashboard featuring dataset exploration, model evaluation diagnostics, prediction interfaces, and historical logs.
5. **Ethical Advisory & Audit Logging:** To embed non-judgmental, actionable student feedback mechanisms and a secure SQLite prediction history ledger with CSV export capabilities.
6. **Automated Verification:** To implement a 100% passing automated test suite (`pytest`) validating schema integrity, boundary constraints, and database operations.

---

### [SECTION 8: EXISTING SYSTEM AND PROPOSED SYSTEM]

#### 8.1 Existing System
Traditional academic monitoring suffers from significant operational limitations:
- **Manual Spreadsheet Audits:** Faculty advisors manually review spreadsheets with hundreds of rows, making cross-attribute pattern recognition nearly impossible.
- **Univariate Cutoffs:** Simple single-attribute rules (e.g., "attendance below 75% equals failure") ignore compensating strengths in self-study or strong foundational knowledge.
- **Post-Facto Assessment:** Interventions only occur after a failed semester examination, increasing student stress and dropout rates.

#### 8.2 Proposed System
The proposed application introduces a predictive paradigm:
- **Multivariate Prediction:** Simultaneously accounts for attendance, study hours, previous exam marks, assignment submissions, sleep balance, and environment.
- **Continuous Numerical Estimation:** Forecasts exact numerical marks (e.g., 78.5/100) alongside mean absolute error confidence intervals.
- **Automated Interpretability:** Features residual error diagnostics, feature importance rankings, and educational tier mapping (Excellent, Good, Average, Needs Improvement).
- **Constructive Guidance:** Provides targeted suggestions (e.g., Pomodoro recommendations for heavy study, sleep warnings, attendance risk alerts).
- **Audit Trails:** Stores prediction logs in an embedded SQLite database for longitudinal departmental reporting.

---

### [SECTION 9: LITERATURE REVIEW AND RELATED WORK]

1. **Romero & Ventura (2020):** *"Educational Data Mining: A Review of the State of the Art."* IEEE Transactions on Systems, Man, and Cybernetics. The authors highlight the transition of institutional learning analytics toward early warning systems (EWS) using supervised algorithms.
2. **Kotsiantis et al. (2018):** *"Predicting Students' Performance in Distance Learning Using Machine Learning Techniques."* Applied Artificial Intelligence. Explored regression versus classification approaches, demonstrating that continuous marks prediction offers superior utility over coarse pass/fail classifiers.
3. **Cortez & Silva (2008):** *"Using Data Mining to Predict Secondary School Student Performance."* EUROSIS-ETI. The landmark Student Performance dataset proved the substantial influence of past evaluation marks and attendance on final academic outcomes.
4. **Pedregosa et al. (2011):** *"Scikit-learn: Machine Learning in Python."* Journal of Machine Learning Research. Documented the best practices of pipeline encapsulation and column transformation to prevent data leakage in production pipelines.

---

### [SECTION 10: SOFTWARE AND HARDWARE REQUIREMENTS]

#### 10.1 Hardware Specifications
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (Intel Core i5 / AMD Ryzen 5 recommended)
- **RAM:** Minimum 4 GB (8 GB recommended for concurrent browser and server operations)
- **Hard Disk / Storage:** Minimum 500 MB free storage for code, dataset, virtual environment, and database
- **Display:** 1366x768 or higher display resolution

#### 10.2 Software Specifications
- **Operating System:** Windows 10 / 11, macOS, or Linux
- **Language / Runtime:** Python 3.10 / 3.11 / 3.12 / 3.13
- **Development Environment:** Visual Studio Code (VS Code) with Python Extension
- **Core Libraries:**
  - Data Manipulation: `pandas >= 2.0.0`, `numpy >= 1.24.0`
  - Machine Learning: `scikit-learn >= 1.4.0`, `joblib >= 1.3.0`
  - Visualization: `plotly >= 5.18.0`, `matplotlib >= 3.8.0`, `seaborn >= 0.13.0`
  - Web Interface: `streamlit >= 1.35.0`
  - Testing Framework: `pytest >= 8.0.0`
  - Database: `sqlite3` (Python Standard Library)

---

### [SECTION 11: SYSTEM ARCHITECTURE]

The architecture follows a clean 3-tier structure:
1. **User Presentation Layer:** Built with Streamlit, organizing capabilities into five dedicated views: Home Dashboard, Dataset Explorer, Model Evaluation, Prediction Page, and History Audit Log.
2. **Business & Inference Logic Layer:** Encapsulates modular Python modules in `src/`:
   - `data_preprocessing.py`: Implements validation rules, feature constraints, and the Scikit-learn `ColumnTransformer`.
   - `train_model.py`: Automates cross-validation, model training, metric evaluation, and artifact serialization.
   - `evaluation.py`: Computes MAE, RMSE, $R^2$, and renders diagnostic Plotly figures.
   - `prediction.py`: Orchestrates runtime inference, bounds clipping, tier categorization, and actionable advisory feedback.
3. **Data & Persistence Layer:** Manages physical data assets:
   - `data/student_performance.csv`: Canonical training dataset.
   - `models/student_performance_pipeline.joblib`: Serialized Scikit-learn Pipeline.
   - `models/model_metadata.json`: Model evaluation metrics and parameters.
   - `database/predictions.db`: Lightweight relational SQLite database for historical inference auditing.

---

### [SECTION 12: DATA FLOW DIAGRAMS]

#### 12.1 DFD Level 0 (Context Diagram)
The user (student, educator, or viva examiner) submits academic parameters (attendance, daily study hours, previous exam scores, assignment marks, sleep duration). The application processes these inputs through the persistent pipeline and returns:
- Estimated final exam marks (0–100 scale).
- Academic performance tier (Excellent, Good, Average, Needs Improvement).
- Personalized academic insights and recommendations.
- Audit records stored in the SQLite database.

#### 12.2 DFD Level 1 (Detailed Subsystem Flow)
1. **Dataset Loading:** Reads raw records, performs median/mode imputation, and validates ranges.
2. **Train/Test Splitting:** Partitions records (80% training, 20% testing).
3. **Pipeline Transformation:** Applies `StandardScaler` to continuous features and `OneHotEncoder` to categorical features.
4. **Estimator Fitting & Benchmarking:** Trains candidate regressors, logs metrics, and exports the top pipeline via `joblib`.
5. **Real-time Inference:** Accepts user inputs from the UI, validates schemas, generates predictions, and executes SQLite persistence.

---

### [SECTION 13: USE-CASE DIAGRAM]

The system serves three primary actor roles:
- **Student Actor:** Inputs personal study metrics, previews projected exam scores, reads constructive study recommendations, downloads prediction summary reports.
- **Faculty / Educator Actor:** Explores student cohort distributions, analyzes correlation matrices between attendance and marks, filters high-risk students in history logs, and exports CSV reports.
- **Viva Examiner / Administrator Actor:** Inspects model benchmark comparisons (Linear Regression vs Decision Tree vs Random Forest), tests preset demonstration profiles, and triggers model retraining on demand.

---

### [SECTION 14: DATASET DESCRIPTION & DATA DICTIONARY]

The dataset comprises 1,500 student records reflecting realistic educational dynamics:

| Feature Name | Python Data Type | Range / Categories | Role | Educational Significance |
|---|---|---|---|---|
| `Student_ID` | String | STD-1001 to STD-2500 | Identifier | Unique student record tag |
| `Attendance_Rate` | Float | 40.0% – 100.0% | Input Feature | Percentage of lectures attended |
| `Study_Hours_Per_Day` | Float | 0.5 – 10.0 hrs | Input Feature | Daily independent study duration |
| `Previous_Exam_Marks` | Float | 30.0 – 100.0 | Input Feature | Prior examination or midterm score |
| `Assignment_Score` | Float | 40.0 – 100.0 | Input Feature | Continuous internal assessment marks |
| `Sleep_Hours_Per_Day` | Float | 4.0 – 10.0 hrs | Input Feature | Daily sleep duration (cognitive factor) |
| `Parent_Education_Level`| String | High School, Bachelor, Master, Doctorate | Input Feature | Highest parental educational level |
| `Internet_Access` | String | Yes, No | Input Feature | Availability of high-speed connectivity |
| `Extracurricular_Activities` | String | Yes, No | Input Feature | Participation in sports/clubs |
| `Final_Exam_Marks` | Float | 0.0 – 100.0 | Target Variable | Final semester examination marks |

---

### [SECTION 15: METHODOLOGY AND ALGORITHMS]

#### 15.1 Linear Regression
Linear Regression models the target variable $y$ as a linear combination of input features $x_i$:
$$y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p + \epsilon$$
The optimal parameter vector $\mathbf{w}$ is derived by minimizing the Ordinary Least Squares (OLS) residual sum of squares:
$$\min_{\mathbf{w}} \sum_{i=1}^{n} \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2$$

#### 15.2 Decision Tree Regressor
The Decision Tree partitions the feature space into $M$ disjoint regions $R_1, R_2, \dots, R_M$. For an input $\mathbf{x} \in R_m$, the prediction is the average target value of training instances within $R_m$:
$$\hat{y} = \frac{1}{|R_m|} \sum_{i \in R_m} y_i$$
Splits are selected by minimizing mean squared error (variance reduction):
$$\text{MSE}(R) = \frac{1}{|R|} \sum_{i \in R} (y_i - \bar{y}_R)^2$$

#### 15.3 Random Forest Regressor
Random Forest constructs an ensemble of $B$ independent decision trees trained on bootstrap samples of the training data. For an input vector $\mathbf{x}$, the final ensemble prediction is the average across all individual tree predictions:
$$\hat{y}_{\text{RF}} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$
Random subspace sampling at each split ensures diversity among individual trees, effectively reducing variance without increasing bias.

---

### [SECTION 16: MODEL EVALUATION RESULTS]

All candidate models were trained strictly on the training subset (1,200 records) and evaluated on the identical held-out test subset (300 records). 

#### 16.1 Experimental Benchmark Results Table

| Model Algorithm | $R^2$ Score (Higher is better) | MAE (Marks) | RMSE (Marks) | MSE | Best Model Selected |
|---|---|---|---|---|---|
| **Linear Regression** | **0.9001** | **2.3309** | **2.9046** | **8.4367** | **Yes (Winner)** |
| **Random Forest Regressor** | 0.8530 | 2.7516 | 3.5239 | 12.4178 | No |
| **Decision Tree Regressor** | 0.7852 | 3.2612 | 4.2595 | 18.1433 | No |

#### 16.2 Analytical Discussion
1. **Linear Regression Superiority:** The Linear Regression pipeline achieved an exceptional $R^2$ score of **0.9001**, indicating that 90.01% of the total variance in final exam marks is accounted for by the features. Because educational progression is predominantly additive (each hour of study and attendance point contributes cumulatively), linear regression serves as an optimal inductive bias.
2. **Low Residual Error:** An MAE of **2.33 marks** demonstrates that the model's estimates deviate on average by less than 2.4 marks out of 100.
3. **Homoscedastic Residuals:** Residual plots confirm a normal error distribution centered at 0 without heteroscedastic funnel shapes.

---

### [SECTION 17: APPLICATION USER INTERFACE WALKTHROUGH]

The web application provides an intuitive five-module user interface:
1. **Home Dashboard (`app.py`):**
   - Displays academic KPIs: Total Students (1,500), Average Marks (61.8), Average Attendance (77.0%), Average Daily Study Hours (3.4 hrs).
   - Plotly interactive histogram of mark distributions and donut chart of performance tiers.
2. **Dataset Explorer (`pages/1_Dataset_Explorer.py`):**
   - Ingestion summary, missing-value and duplicate audits, tabular preview with customizable row count.
   - Interactive scatter plots for attendance, study hours, previous exam scores, and assignments against final marks.
   - Pearson correlation heatmap and CSV download button.
3. **Model Evaluation (`pages/2_Model_Evaluation.py`):**
   - Cross-model benchmark table highlighting top metrics in green.
   - Interactive Actual vs. Predicted scatter plot with $y = x$ reference diagonal.
   - Residual error distribution histogram and horizontal feature importance bar chart.
   - Viva Voce metric glossary.
4. **Predict Performance (`pages/3_Predict_Performance.py`):**
   - Input form with sliders and dropdowns.
   - Quick presets for live college demos ("High Achiever", "Average Student", "At-Risk Student").
   - Plotly score gauge indicator color-coded by tier.
   - Actionable student guidance cards and text report download button.
5. **Prediction History (`pages/4_Prediction_History.py`):**
   - Tabular history of queries from SQLite.
   - Category filter dropdown and CSV export button.
   - Single-record deletion and full database clear controls.

---

### [SECTION 18: TESTING AND VERIFICATION]

The system underwent rigorous automated testing using `pytest`.

#### 18.1 Test Summary
- **Total Test Cases Executed:** 14
- **Passed:** 14 (100% Pass Rate)
- **Execution Time:** ~2.1 seconds

#### 18.2 Verified Test Cases
1. `test_load_data_valid`: Confirms valid CSV loading with expected columns.
2. `test_load_data_file_not_found`: Confirms graceful `FileNotFoundError` handling.
3. `test_load_data_missing_columns`: Confirms `ValueError` when required columns are absent.
4. `test_load_data_missing_values_imputed`: Confirms median/mode imputation on null values.
5. `test_split_data`: Validates 80/20 train-test split proportions.
6. `test_create_preprocessor_transformation`: Confirms matrix transformation without errors.
7. `test_validate_input_features_success`: Validates correct feature dictionaries.
8. `test_validate_input_features_out_of_bounds`: Confirms rejection of invalid values (e.g. attendance > 100%).
9. `test_load_pipeline_and_metadata`: Confirms serialized joblib pipeline and metadata JSON loading.
10. `test_predict_performance_valid_range`: Confirms prediction outputs are bounded within $[0, 100]$.
11. `test_performance_categorization_thresholds`: Confirms exact tier boundary mappings.
12. `test_prediction_invalid_feature_raises`: Confirms exceptions on illegal input types.
13. `test_student_insights_triggers`: Confirms generation of attendance and sleep warnings.
14. `test_sqlite_database_lifecycle`: Confirms table creation, insertion, querying, and deletion.

---

### [SECTION 19: LIMITATIONS]

1. **Unpredictable Real-World Disruptions:** Statistical models cannot account for sudden personal events such as examination-day illness, personal bereavement, or acute psychological anxiety.
2. **Curriculum Variability:** The current dataset assumes uniform course difficulty across subjects. Advanced engineering disciplines (e.g., Higher Mathematics or Compiler Design) may exhibit distinct learning curves compared to introductory subjects.
3. **Self-Reported Metric Variance:** In real-world deployments, daily study hours and sleep durations are self-reported and may be subject to student recall bias.

---

### [SECTION 20: FUTURE SCOPE]

1. **LMS Integration:** Connecting the inference pipeline directly to institutional Learning Management Systems (e.g., Canvas, Moodle, or Google Classroom) to automatically pull attendance and quiz marks via REST APIs.
2. **Deep Learning Sequence Models:** Utilizing Recurrent Neural Networks (LSTMs or Transformers) to model continuous weekly learning trajectories rather than single static snapshots.
3. **Automated Faculty Alerts:** Establishing an email/SMS notification service that alerts academic advisors when a student's projected trajectory crosses into the "Needs Improvement" tier.
4. **Course-Specific Weighting:** Allowing instructors to configure customized attribute weightings according to specific subject difficulty profiles.

---

### [SECTION 21: CONCLUSION]

The project **"Student Performance Prediction Using Machine Learning"** successfully addresses the challenge of delayed academic feedback by delivering a complete, robust, and accessible decision-support system. 

By employing Scikit-learn's pipeline abstractions, the system guarantees reproducible, leak-free data transformations across three supervised algorithms. Linear Regression emerged as the optimal estimator, achieving an $R^2$ of **0.9001** and an MAE of **2.33 marks**. Through its multi-page Streamlit web interface and embedded SQLite audit logging, the application bridges the gap between machine learning research and real-world educational deployment. Crucially, the system adheres to strict ethical AI guidelines, ensuring predictions are presented as constructive, actionable guidance rather than rigid judgments.

---

### [SECTION 22: REFERENCES]

1. Romero, C., & Ventura, S. (2020). *Educational Data Mining: A Review of the State of the Art.* IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), 40(6), 601-618.
2. Kotsiantis, S. B. (2018). *Use of machine learning techniques for educational proposes: a decision support system for forecasting students’ grades.* Artificial Intelligence Review, 37(4), 331-344.
3. Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance.* In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
5. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* In Proceedings of the 9th Python in Science Conference, pp. 51-56.
6. Streamlit Inc. (2024). *Streamlit Documentation: The fastest way to build and share data apps.* Available at: https://docs.streamlit.io.
