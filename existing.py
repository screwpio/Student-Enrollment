import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from functools import lru_cache

# Load and clean dataset
df = pd.read_csv("Students List(Student List).csv")
df = df[df["Student ID"].notnull()]
df["Student ID"] = df["Student ID"].astype(float).astype(int)
df["Major Applied for"] = df["Major Applied for"].astype(str).str.strip()
df["Gender"] = df["Gender"].astype(str).str.strip()
df["Course"] = df["Course"].astype(str).str.strip()
df["SUBJECT"] = df["SUBJECT"].astype(str).str.strip()
df["Course Title"] = df["Course Title"].astype(str).str.strip()
df["Age When Applied"] = pd.to_numeric(df["Age When Applied"], errors='coerce')
df["Number"] = pd.to_numeric(df["Number"], errors='coerce')

# Preloaded mappings
course_title_map = df.drop_duplicates("Course").set_index("Course")["Course Title"].to_dict()
course_info = df.drop_duplicates("Course").set_index("Course")[["SUBJECT", "Number"]].to_dict(orient="index")
major_subject_map = df.groupby(["Major Applied for", "SUBJECT"]).size().unstack(fill_value=0)

# Cache similar students by major
@lru_cache(maxsize=None)
def get_similar_students(major):
    return df[df["Major Applied for"] == major]

# Vectorized similarity
def compute_similarity(sim_df, gender, age):
    sim = np.ones(len(sim_df))
    sim *= np.where(sim_df["Gender"] == gender, 1.0, 0.7)
    if not np.isnan(age):
        sim *= np.maximum(0.5, 1 - np.abs(sim_df["Age When Applied"] - age) / 10)
    return sim

# Recommender function
def recommend_courses(student_id, top_n=3):
    if student_id not in df["Student ID"].values:
        return []

    student_df = df[df["Student ID"] == student_id]
    student_info = student_df.iloc[0]
    major = student_info["Major Applied for"]
    gender = student_info["Gender"]
    age = student_info["Age When Applied"]
    taken_courses = set(student_df["Course"])
    taken_subjects = set(student_df["SUBJECT"])
    subject_counts = student_df["SUBJECT"].value_counts(normalize=True).to_dict()

    sim_students = get_similar_students(major)
    sim_scores = compute_similarity(sim_students, gender, age)
    sim_students = sim_students.copy()
    sim_students["similarity"] = sim_scores

    course_scores = defaultdict(float)
    course_counts = sim_students.groupby("Course")["similarity"].sum()
    course_totals = sim_students["Student ID"].nunique()

    for course, score in course_counts.items():
        if course in taken_courses or course not in course_info:
            continue

        subject = course_info[course]["SUBJECT"]
        level = course_info[course]["Number"]
        subject_pref = subject_counts.get(subject, 0.1)
        subject_weight = major_subject_map.loc[major].get(subject, 0.1)
        avg_taken_level = student_df[student_df["SUBJECT"] == subject]["Number"].mean()
        level_boost = 1.0

        if not pd.isna(avg_taken_level) and not pd.isna(level):
            if level > avg_taken_level:
                level_boost = 1.2
            elif level < avg_taken_level:
                level_boost = 0.8

        norm_score = score / course_totals
        final_score = norm_score * subject_pref * subject_weight * level_boost
        course_scores[course] = final_score

    if len(course_scores) < top_n:
        related_students = df[df["SUBJECT"].isin(taken_subjects)]
        fallback_courses = Counter(related_students["Course"])
        fallback_courses = {k: v for k, v in fallback_courses.items() if k not in taken_courses and k not in course_scores}
        for course, count in fallback_courses.items():
            course_scores[course] += count / 100

    top_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = [(student_id, course, course_title_map.get(course, "Unknown Title"), round(score, 3)) for course, score in top_courses]
    return result

# Run recommender for all students
all_results = []
unique_student_ids = df["Student ID"].unique()

for student_id in tqdm(unique_student_ids, desc="Generating Recommendations"):
    all_results.extend(recommend_courses(student_id))

# Save results
results_df = pd.DataFrame(all_results, columns=["Student ID", "Predicted Course", "Course Title", "Confidence"])
results_df.to_csv("student_predictions.csv", index=False)

print("Saved predictions to student_predictions.csv")
