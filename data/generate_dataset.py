"""
Student Performance Dataset Generator
-------------------------------------
Generates a realistic, statistically coherent academic dataset for training
and evaluating machine learning models to predict final examination scores.

Dataset characteristics:
- 1,500 student records
- Features: Attendance, Study Hours, Previous Exam Marks, Assignment Scores,
  Sleep Hours, Extracurricular Activities, Parental Education, Internet Access.
- Target: Final Exam Marks (0 - 100 scale)
- Deterministic random seed for 100% reproducibility.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_student_dataset(n_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """Generate synthetic academic performance dataset with realistic relationships."""
    np.random.seed(random_seed)

    # 1. Student IDs
    student_ids = [f"STD-{1001 + i}" for i in range(n_samples)]

    # 2. Previous Exam Marks (Mean ~ 68, SD ~ 14, bounded [30, 98])
    prev_marks = np.random.normal(loc=68.0, scale=13.5, size=n_samples)
    prev_marks = np.clip(prev_marks, 30.0, 98.0)

    # 3. Attendance Rate (Mean ~ 80%, bounded [45, 99.5])
    # Higher previous achievers tend to attend classes slightly more regularly
    attendance = 0.6 * np.random.normal(loc=78.0, scale=12.0, size=n_samples) + 0.4 * (prev_marks * 0.95 + 10)
    attendance = np.clip(attendance, 45.0, 99.5)

    # 4. Daily Study Hours (Mean ~ 3.6 hours, bounded [0.5, 9.5])
    study_hours = 0.5 * np.random.gamma(shape=3.5, scale=1.0, size=n_samples) + 0.02 * attendance
    study_hours = np.clip(study_hours, 0.5, 9.5)

    # 5. Assignment Score (Mean ~ 73, bounded [40, 100])
    assignment_score = 0.45 * prev_marks + 0.35 * attendance + np.random.normal(loc=12.0, scale=7.0, size=n_samples)
    assignment_score = np.clip(assignment_score, 40.0, 100.0)

    # 6. Sleep Hours (Mean ~ 7.0 hours, bounded [4.5, 9.5])
    # Extreme study hours often correlate with slight sleep reduction
    sleep_hours = np.random.normal(loc=7.2, scale=1.1, size=n_samples) - 0.12 * (study_hours - 3.5)
    sleep_hours = np.clip(sleep_hours, 4.5, 9.5)

    # 7. Categorical features
    parent_education_choices = ["High School", "Bachelor", "Master", "Doctorate"]
    parent_education_probs = [0.28, 0.46, 0.20, 0.06]
    parent_edu = np.random.choice(parent_education_choices, size=n_samples, p=parent_education_probs)

    internet_choices = ["Yes", "No"]
    internet_probs = [0.88, 0.12]
    internet_access = np.random.choice(internet_choices, size=n_samples, p=internet_probs)

    extracurricular_choices = ["Yes", "No"]
    extracurricular_probs = [0.48, 0.52]
    extracurricular = np.random.choice(extracurricular_choices, size=n_samples, p=extracurricular_probs)

    # 8. Compute realistic Final Exam Marks
    # Base formula reflecting empirical pedagogical research:
    # Final = w1*Prev + w2*Attendance + w3*StudyHours + w4*Assignment + SleepBonus + ContextBonus + Noise
    study_effect = 2.8 * study_hours - 0.12 * np.maximum(0, study_hours - 6.5) ** 2  # Diminishing returns after 6.5 hrs
    attendance_effect = 0.24 * attendance
    prev_effect = 0.34 * prev_marks
    assignment_effect = 0.18 * assignment_score

    # Sleep quality effect: optimum between 6.5 and 8.0 hours
    sleep_effect = np.where(
        sleep_hours < 6.0,
        -1.8 * (6.0 - sleep_hours),
        np.where(sleep_hours > 8.5, -0.6 * (sleep_hours - 8.5), 1.2)
    )

    # Contextual effect
    parent_bonus = pd.Series(parent_edu).map({"High School": 0.0, "Bachelor": 0.8, "Master": 1.4, "Doctorate": 1.9}).values
    internet_bonus = np.where(internet_access == "Yes", 0.8, -1.2)
    extra_bonus = np.where(extracurricular == "Yes", 0.4, 0.0)

    # Realistic random noise (evaluation day variability)
    random_noise = np.random.normal(loc=0.0, scale=2.8, size=n_samples)

    final_marks = (
        prev_effect
        + attendance_effect
        + study_effect
        + assignment_effect
        + sleep_effect
        + parent_bonus
        + internet_bonus
        + extra_bonus
        + random_noise
        - 4.0  # calibration intercept
    )

    final_marks = np.clip(final_marks, 25.0, 100.0)

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Attendance_Rate": np.round(attendance, 1),
        "Study_Hours_Per_Day": np.round(study_hours, 1),
        "Previous_Exam_Marks": np.round(prev_marks, 1),
        "Assignment_Score": np.round(assignment_score, 1),
        "Sleep_Hours_Per_Day": np.round(sleep_hours, 1),
        "Parent_Education_Level": parent_edu,
        "Internet_Access": internet_access,
        "Extracurricular_Activities": extracurricular,
        "Final_Exam_Marks": np.round(final_marks, 1)
    })

    return df


def main():
    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / "student_performance.csv"
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Generating student performance dataset (1,500 samples)...")
    df = generate_student_dataset(n_samples=1500, random_seed=42)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created and saved to: {output_path}")
    print("\nDataset Summary Statistics:")
    print(df.describe())
    print("\nFirst 5 records:")
    print(df.head())


if __name__ == "__main__":
    main()
