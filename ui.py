import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import joblib

# === Load data ===
df_info = pd.read_csv("Students List(Student List).csv")
df_info["Student ID"] = pd.to_numeric(df_info["Student ID"], errors="coerce")
df_info = df_info[df_info["Student ID"].notnull()]
df_info["Student ID"] = df_info["Student ID"].astype(int)

df_preds = pd.read_csv("student_predictions.csv")
model = joblib.load("course_model.pkl")
mlb = joblib.load("mlb.pkl")
input_columns = joblib.load("input_columns.pkl")
course_title_map = joblib.load("course_title_map.pkl")

# === Predict Function ===
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

# === Setup GUI ===
root = tk.Tk()
root.title("Course Prediction System")
root.geometry("720x600")
root.configure(bg="#1e1e1e")

style = ttk.Style()
style.theme_use("default")
style.configure(".", foreground="white", background="#2d2d2d", font=("Segoe UI", 10))
style.configure("TEntry", fieldbackground="#3c3c3c", foreground="white")
style.configure("TButton", background="#444", foreground="white")
style.configure("TCombobox", fieldbackground="#3c3c3c", background="#3c3c3c", foreground="white")
style.configure("TNotebook", background="#2d2d2d")
style.configure("TNotebook.Tab", background="#3c3c3c", padding=[10, 3])
style.map("TNotebook.Tab", background=[("selected", "#1f1f1f")])

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=10, pady=10)

majors = sorted(df_info["Major Applied for"].dropna().unique())
genders = sorted(df_info["Gender"].dropna().unique())

# ========== Existing Student ==========
frame_existing = ttk.Frame(notebook)
notebook.add(frame_existing, text="Existing Student")

tk.Label(frame_existing, text="Student ID:").pack()
student_id_var = tk.StringVar()
tk.Entry(frame_existing, textvariable=student_id_var, width=25).pack()
existing_output_frame = ttk.Frame(frame_existing)
existing_output_frame.pack(pady=10, fill="x")

def clear_existing():
    for widget in existing_output_frame.winfo_children():
        widget.destroy()

def show_existing():
    clear_existing()
    try:
        sid = int(student_id_var.get())
        student = df_info[df_info["Student ID"] == sid]
        taken = student[["Course", "Course Title"]].drop_duplicates()
        preds = df_preds[df_preds["Student ID"] == sid]

        if student.empty:
            ttk.Label(existing_output_frame, text="Student not found.").pack()
            return

        s = student.iloc[0]
        ttk.Label(existing_output_frame, text=f"Gender: {s['Gender']}").pack(anchor="w")
        ttk.Label(existing_output_frame, text=f"Major: {s['Major Applied for']}").pack(anchor="w")
        ttk.Label(existing_output_frame, text=f"Age: {s['Age When Applied']}").pack(anchor="w")

        ttk.Label(existing_output_frame, text="Courses Taken:").pack(anchor="w", pady=(8, 0))
        for _, row in taken.iterrows():
            ttk.Label(existing_output_frame, text=f"- {row['Course']}: {row['Course Title']}").pack(anchor="w")

        ttk.Label(existing_output_frame, text="Predicted Courses:").pack(anchor="w", pady=(8, 0))
        for _, row in preds.iterrows():
            ttk.Label(existing_output_frame, text=f"- {row['Predicted Course']}: {row['Course Title']}").pack(anchor="w")

    except Exception as e:
        ttk.Label(existing_output_frame, text=f"Error: {e}").pack()

ttk.Button(frame_existing, text="Show Info", command=show_existing).pack(pady=5)

# ========== New Student ==========
frame_new = ttk.Frame(notebook)
notebook.add(frame_new, text="New Student")

age_var = tk.StringVar()
gender_var = tk.StringVar()
major_var = tk.StringVar()
new_output_frame = ttk.Frame(frame_new)

ttk.Label(frame_new, text="Age:").pack()
ttk.Entry(frame_new, textvariable=age_var, width=25).pack()
ttk.Label(frame_new, text="Gender:").pack()
ttk.Combobox(frame_new, textvariable=gender_var, values=genders, width=22).pack()
ttk.Label(frame_new, text="Major:").pack()
ttk.Combobox(frame_new, textvariable=major_var, values=majors, width=22).pack()

def clear_new_output():
    for widget in new_output_frame.winfo_children():
        widget.destroy()

def predict_new():
    clear_new_output()
    try:
        age = float(age_var.get())
        gender = gender_var.get()
        major = major_var.get()
        predictions = predict_for_new_student(age, gender, major)

        ttk.Label(new_output_frame, text="Predicted Courses:").pack(anchor="w", pady=(8, 0))
        for course, title in predictions:
            ttk.Label(new_output_frame, text=f"- {course}: {title}").pack(anchor="w")

    except Exception as e:
        ttk.Label(new_output_frame, text=f"Error: {e}").pack()

ttk.Button(frame_new, text="Predict", command=predict_new).pack(pady=8)
new_output_frame.pack(pady=5, fill="x")

# ========== Admin ==========
frame_admin = ttk.Frame(notebook)
notebook.add(frame_admin, text="Admin")

admin_canvas = tk.Canvas(frame_admin, bg="#1e1e1e", highlightthickness=0)
admin_scrollbar = ttk.Scrollbar(frame_admin, orient="vertical", command=admin_canvas.yview)
admin_scrollable_frame = ttk.Frame(admin_canvas)

admin_scrollable_frame.bind(
    "<Configure>",
    lambda e: admin_canvas.configure(scrollregion=admin_canvas.bbox("all"))
)

admin_canvas.create_window((0, 0), window=admin_scrollable_frame, anchor="nw")
admin_canvas.configure(yscrollcommand=admin_scrollbar.set)

admin_canvas.pack(side="left", fill="both", expand=True)
admin_scrollbar.pack(side="right", fill="y")

def load_admin_data():
    for widget in admin_scrollable_frame.winfo_children():
        widget.destroy()

    course_counts = df_preds["Predicted Course"].value_counts().reset_index()
    course_counts.columns = ["Course", "Predicted Count"]
    course_counts = course_counts.sort_values(by="Predicted Count", ascending=False)

    for _, row in course_counts.iterrows():
        title = course_title_map.get(row['Course'], "Unknown Title")
        ttk.Label(admin_scrollable_frame, text=f"{row['Course']} - {title}: {row['Predicted Count']} students").pack(anchor="w", pady=2)

ttk.Button(frame_admin, text="Load Predictions", command=load_admin_data).pack(pady=5)

# Run GUI
root.mainloop()
