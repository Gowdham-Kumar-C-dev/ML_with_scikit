from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained ML model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Loads your dashboard UI

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert input to the correct order expected by your ML model
    features = np.array([[
        data["Age"],
        data["Gender"],
        data["Weight"],
        data["Height"],
        data["BMI"],
        data["Smoking"],
        data["Alcohol_Intake"],
        data["Physical_Activity"],
        data["Diet"],
        data["Stress_Level"],
        data["Hypertension"],
        data["Diabetes"],
        data["Hyperlipidemia"],
        data["Family_History"],
        data["Previous_Heart_Attack"],
        data["Systolic_BP"],
        data["Diastolic_BP"],
        data["Heart_Rate"],
        data["Blood_Sugar_Fasting"],
        data["Cholesterol_Total"]
    ]])

    # Predict using the model
    prediction = model.predict(features)[0]

    # Convert numeric result to readable label
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
