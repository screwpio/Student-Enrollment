import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load necessary files and models
df_info = pd.read_csv("Students List(Student List).csv")
df_preds = pd.read_csv("student_predictions.csv")
model = joblib.load("course_model.pkl")
mlb = joblib.load("mlb.pkl")
input_columns = joblib.load("input_columns.pkl")
course_title_map = joblib.load("course_title_map.pkl")

# Prediction function for new student
def predict_for_new_student(age, gender, major, top_n=3):
    new_input = pd.DataFrame([{
        "Age When Applied": age,
        "Gender": gender,
        "Major Applied for": major
    }])
    new_input = pd.get_dummies(new_input).reindex(columns=input_columns, fill_value=0)

    y_pred = model.predict_proba(new_input)
    scores = np.array([probs[:, 1] if probs.shape[1] == 2 else np.zeros(probs.shape[0]) for probs in y_pred]).flatten()
    top_indices = np.argsort(scores)[-top_n:][::-1]
    predictions = [(mlb.classes_[i], course_title_map.get(mlb.classes_[i], "Unknown Title")) for i in top_indices]
    return predictions

# Streamlit interface
st.title("Course Enrollment System")

# Tabs
option = st.sidebar.selectbox("Select User Type", ["Existing Student", "New Student", "Admin"])

if option == "Existing Student":
    st.header("Existing Student Information")
    student_id = st.text_input("Enter Student ID")
    if st.button("Show Info"):
        try:
            sid = int(student_id)
            student = df_info[df_info["Student ID"] == sid]
            preds = df_preds[df_preds["Student ID"] == sid]

            if student.empty:
                st.error("Student not found.")
            else:
                s = student.iloc[0]
                st.write(f"**Gender:** {s['Gender']}")
                st.write(f"**Major:** {s['Major Applied for']}")
                st.write(f"**Age:** {s['Age When Applied']}")
                
                st.subheader("Courses Taken")
                taken_courses = student[["Course", "Course Title"]].drop_duplicates()
                for _, row in taken_courses.iterrows():
                    st.write(f"- {row['Course']}: {row['Course Title']}")
                
                st.subheader("Recommended Courses")
                for _, row in preds.iterrows():
                    st.write(f"- {row['Predicted Course']}: {row['Course Title']}")

        except Exception as e:
            st.error(f"Error: {e}")

elif option == "New Student":
    st.header("New Student Course Recommendations")
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("Gender", df_info["Gender"].dropna().unique())
    major = st.selectbox("Major", df_info["Major Applied for"].dropna().unique())

    if st.button("Predict"):
        predictions = predict_for_new_student(age, gender, major)
        st.subheader("Predicted Courses")
        for course, title in predictions:
            st.write(f"- {course}: {title}")

elif option == "Admin":
    st.header("Admin Course Demand Overview")
    course_counts = df_preds["Predicted Course"].value_counts().reset_index()
    course_counts.columns = ["Course", "Predicted Count"]
    course_counts["Course Title"] = course_counts["Course"].map(course_title_map)
    course_counts_sorted = course_counts.sort_values(by="Predicted Count", ascending=False)

    st.write(course_counts_sorted)

# Run this Streamlit app by saving it as app.py and running 'streamlit run app.py'
