import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

# Load saved artifacts
model = joblib.load(open('model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))
imputer = joblib.load(open('imputer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from HTML form
        features = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter valid numeric values.")

    # Convert to DataFrame for column-wise operations
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([features], columns=columns)

    # Replace zeros with NaN for medically invalid fields
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    input_df[zero_invalid_cols] = input_df[zero_invalid_cols].replace(0, np.nan)

    # Apply imputation and scaling
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    # Predict
    prediction = model.predict(input_scaled)[0]
    confidence = max(model.predict_proba(input_scaled)[0]) * 100

    # Output
    result = f"{'Diabetic' if prediction == 1 else 'Non-Diabetic'} (Model Confidence: {confidence:.2f}%)"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
