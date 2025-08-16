import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

MODEL_FILE = "heartd.pkl"
DATA_FILE = "heart.csv"

# ✅ Train model if not exists
if os.path.exists(MODEL_FILE):
    heart = pickle.load(open(MODEL_FILE, "rb"))
else:
    st.warning("⚠️ Model not found, training a new one...")
    df = pd.read_csv(DATA_FILE)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    heart = DecisionTreeClassifier().fit(X_train, y_train)
    pickle.dump(heart, open(MODEL_FILE, "wb"))
    st.success("✅ Model trained and saved as heartd.pkl")

st.title("❤️ Heart Disease Prediction App")

menu = ["Predict", "Upload CSV"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Predict":
    st.subheader("Enter Patient Details")
    
    # Assuming features from dataset
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.number_input("Oldpeak", value=1.0)
    slope = st.selectbox("Slope (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1=normal, 2=fixed defect, 3=reversible defect)", [1, 2, 3])

    if st.button("Predict"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                              exang, oldpeak, slope, ca, thal]])
        prediction = heart.predict(features)[0]
        result = "💔 Heart Disease Detected" if prediction == 1 else "❤️ No Heart Disease"
        st.success(result)

elif choice == "Upload CSV":
    st.subheader("Batch Prediction with CSV")
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Uploaded Data:", df.head())
        preds = heart.predict(df)
        df["Prediction"] = ["💔" if p == 1 else "❤️" for p in preds]
        st.write(df)
        st.download_button("Download Results", df.to_csv(index=False), "predictions.csv")
