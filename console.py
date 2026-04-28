import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
# DATASET 

np.random.seed(42)
data_size = 200

study_hours = np.random.randint(1, 10, data_size)
prev_marks = np.random.randint(30, 100, data_size)
play_hours = np.random.randint(0, 6, data_size)
attendance = np.random.randint(50, 100, data_size)
assignments = np.random.randint(0, 10, data_size)

final_score = (
    study_hours * 5 +
    prev_marks * 0.5 -
    play_hours * 2 +
    attendance * 0.2 +
    assignments * 2 +
    np.random.randint(-5,5,data_size)
)

df = pd.DataFrame({
    "study_hours": study_hours,
    "prev_marks": prev_marks,
    "play_hours": play_hours,
    "attendance": attendance,
    "assignments": assignments,
    "final_score": final_score
})

# Regression Model
X = df.drop("final_score", axis=1)
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Classification Model
df["pass_fail"] = df["final_score"].apply(lambda x: 1 if x >= 50 else 0)

X2 = df.drop(["final_score", "pass_fail"], axis=1)
y2 = df["pass_fail"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

clf_model = DecisionTreeClassifier()
clf_model.fit(X_train2, y_train2)
# FUNCTIONS

file_name = "student_history.csv"

def predict_student():
    print("\n====== ENTER STUDENT DETAILS ======")

    study = float(input("Study Hours per day: "))
    prev = float(input("Previous Marks: "))
    play = float(input("Playing Hours per day: "))
    att = float(input("Attendance Percentage: "))
    ass = float(input("Assignments Completed: "))

    user_df = pd.DataFrame({
        "study_hours": [study],
        "prev_marks": [prev],
        "play_hours": [play],
        "attendance": [att],
        "assignments": [ass]
    })

    score_pred = reg_model.predict(user_df)[0]
    pass_pred = clf_model.predict(user_df)[0]

    result = "PASS" if pass_pred == 1 else "FAIL"

    print("\nPredicted Final Score:", round(score_pred,2))
    print("Result:", result)

    # Save data
    user_df["predicted_score"] = round(score_pred,2)
    user_df["result"] = result

    if os.path.exists(file_name):
        user_df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        user_df.to_csv(file_name, mode='w', header=True, index=False)

    print("✅ Student data saved!")


def view_all():
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        print("\n=== ALL STUDENT RECORDS ===")
        print(df)
    else:
        print("No data found.")


def view_last():
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        print("\n=== LAST STUDENT RECORD ===")
        print(df.tail(1))
    else:
        print("No data found.")


def statistics():
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        print("\n=== STATISTICS ===")
        print("Total Students:", len(df))
        print("Average Score:", df["predicted_score"].mean())
        print("Pass Count:", (df["result"] == "PASS").sum())
        print("Fail Count:", (df["result"] == "FAIL").sum())
    else:
        print("No data found.")


def search_student():
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        hrs = float(input("Show students with study hours greater than: "))
        result = df[df["study_hours"] > hrs]
        print(result)
    else:
        print("No data found.")
# MENU
while True:
    print("\n====== STUDENT MENU ======")
    print("1. Predict Student Performance")
    print("2. View All Students")
    print("3. View Last Student")
    print("4. Show Statistics")
    print("5. Search Students")
    print("6. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        predict_student()
    elif choice == "2":
        view_all()
    elif choice == "3":
        view_last()
    elif choice == "4":
        statistics()
    elif choice == "5":
        search_student()
    elif choice == "6":
        print("Exiting Program...")
        break
    else:
        print("Invalid Choice")