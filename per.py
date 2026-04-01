import customtkinter as ctk
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# UI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1300x750")
app.title("StudentAI Dashboard")

# FILES
USERS_FILE = "users.csv"
PROFILES_FILE = "profiles.csv"
current_user = None

if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USERS_FILE, index=False)

if not os.path.exists(PROFILES_FILE):
    pd.DataFrame(columns=["username", "name", "class", "target", "daily_goal"]).to_csv(PROFILES_FILE, index=False)

# MODEL
data = pd.read_csv("StudentsPerformance.csv")

data["avg_score"] = (data["math score"] + data["reading score"] + data["writing score"]) / 3
data["study"] = np.random.randint(1, 6, len(data))
data["play"] = np.random.randint(1, 6, len(data))
data["attendance"] = np.random.randint(60, 100, len(data))
data["assignments"] = np.random.randint(1, 10, len(data))

X = data[["study", "play", "attendance", "assignments"]]
y_reg = data["avg_score"]
y_clf = (data["avg_score"] >= 40).astype(int)

reg_model = LinearRegression().fit(X, y_reg)
clf_model = DecisionTreeClassifier().fit(X, y_clf)

# LOGIN
def show_login():
    for w in app.winfo_children():
        w.destroy()

    frame = ctk.CTkFrame(app)
    frame.pack(expand=True)

    ctk.CTkLabel(frame, text="Login", font=("Arial", 28)).pack(pady=20)

    user = ctk.CTkEntry(frame, placeholder_text="Username")
    user.pack(pady=10)

    pwd = ctk.CTkEntry(frame, placeholder_text="Password", show="*")
    pwd.pack(pady=10)

    status = ctk.StringVar()

    def login():
        global current_user
        df = pd.read_csv(USERS_FILE)

        match = df[(df["username"] == user.get()) & (df["password"] == pwd.get())]

        if not match.empty:
            current_user = user.get()
            build_app()
        else:
            status.set("Invalid Login")

    def signup():
        df = pd.read_csv(USERS_FILE)

        if user.get() in df["username"].values:
            status.set("User exists")
            return

        pd.DataFrame([{
            "username": user.get(),
            "password": pwd.get()
        }]).to_csv(USERS_FILE, mode="a", header=False, index=False)

        status.set("Signup success")

    ctk.CTkButton(frame, text="Login", command=login).pack(pady=5)
    ctk.CTkButton(frame, text="Signup", command=signup).pack(pady=5)
    ctk.CTkLabel(frame, textvariable=status).pack()

# MAIN
def build_app():
    for w in app.winfo_children():
        w.destroy()

    sidebar = ctk.CTkFrame(app, width=220)
    sidebar.pack(side="left", fill="y")

    main = ctk.CTkFrame(app)
    main.pack(side="right", expand=True, fill="both")

    def clear():
        for w in main.winfo_children():
            w.destroy()

    # SAVE
    def save(values, score, result):
        file = f"records_{current_user}.csv"

        df = pd.DataFrame([{
            "date": datetime.datetime.now(),
            "study": values[0],
            "play": values[1],
            "attendance": values[2],
            "assignments": values[3],
            "score": score,
            "result": "PASS" if result else "FAIL"
        }])

        df.to_csv(file, mode="a", header=not os.path.exists(file), index=False)

    # 🔥 ADVISOR WITH GOALS
    def advisor(df):
        advice = []

        avg_study = df["study"].mean()
        avg_score = df["score"].mean()

        profiles = pd.read_csv(PROFILES_FILE)
        user_data = profiles[profiles["username"] == current_user]

        target = None
        daily_goal = None

        if not user_data.empty:
            try:
                target = float(user_data.iloc[0]["target"])
            except:
                pass
            try:
                daily_goal = float(user_data.iloc[0]["daily_goal"])
            except:
                pass

        if daily_goal:
            percent = (avg_study / daily_goal) * 100
            advice.append(f"Goal: {daily_goal} hrs | {percent:.1f}%")

            if percent >= 100:
                advice.append("Goal Achieved")
            elif percent >= 70:
                advice.append("On Track")
            else:
                advice.append("Behind")

        if target:
            gap = target - avg_score
            advice.append(f"\nTarget: {target}")

            if gap > 10:
                advice.append("Far from target")
            elif gap > 0:
                advice.append("Close to target")
            else:
                advice.append("Target achieved")

        advice.append(f"\nAvg Score: {avg_score:.2f}")

        improved = reg_model.predict([[5,2,90,8]])[0]
        advice.append(f"If 5h study → {improved:.2f}")

        return "\n".join(advice)

    # PROFILE
    def profile_page():
        clear()

        df = pd.read_csv(PROFILES_FILE)
        user_data = df[df["username"] == current_user]

        frame = ctk.CTkFrame(main)
        frame.pack(pady=40)

        name = ctk.CTkEntry(frame, placeholder_text="Name")
        name.pack()

        cls = ctk.CTkEntry(frame, placeholder_text="Class")
        cls.pack()

        target = ctk.CTkEntry(frame, placeholder_text="Target Score")
        target.pack()

        goal = ctk.CTkEntry(frame, placeholder_text="Daily Study Goal")
        goal.pack()

        if not user_data.empty:
            name.insert(0, user_data.iloc[0]["name"])
            cls.insert(0, user_data.iloc[0]["class"])
            target.insert(0, user_data.iloc[0]["target"])
            goal.insert(0, user_data.iloc[0].get("daily_goal", ""))

        def save_profile():
            nonlocal df

            df = df[df["username"] != current_user]

            new = pd.DataFrame([{
                "username": current_user,
                "name": name.get(),
                "class": cls.get(),
                "target": target.get(),
                "daily_goal": goal.get()
            }])

            df = pd.concat([df, new])
            df.to_csv(PROFILES_FILE, index=False)

        ctk.CTkButton(frame, text="Save", command=save_profile).pack(pady=10)

    # WEEKLY
    def weekly_page():
        clear()

        file = f"records_{current_user}.csv"

        if not os.path.exists(file):
            return

        df = pd.read_csv(file).tail(7)

        # graph
        fig, ax = plt.subplots()
        ax.plot(df["score"], marker="o")

        canvas = FigureCanvasTkAgg(fig, master=main)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # goal progress
        profiles = pd.read_csv(PROFILES_FILE)
        user_data = profiles[profiles["username"] == current_user]

        if not user_data.empty:
            try:
                goal = float(user_data.iloc[0]["daily_goal"])
                avg = df["study"].mean()

                percent = avg / goal

                bar = ctk.CTkProgressBar(main)
                bar.pack(pady=10)
                bar.set(min(percent, 1))

            except:
                pass

        box = ctk.CTkTextbox(main, width=800, height=200)
        box.pack()
        box.insert("end", advisor(df))

    # SIDEBAR
    ctk.CTkButton(sidebar, text="Predict", command=lambda: clear()).pack(pady=10)
    ctk.CTkButton(sidebar, text="Weekly", command=weekly_page).pack(pady=10)
    ctk.CTkButton(sidebar, text="Profile", command=profile_page).pack(pady=10)

# START
show_login()
app.mainloop()