import numpy as np
import joblib
from flask import Flask, render_template, request

# Load saved artifacts
model = joblib.load(open('model.pkl','rb'))
scaler = joblib.load(open('scaler.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from HTML form
    try:
        features = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter valid numeric values.")

    # Convert to array
    input_data = np.array(features).reshape(1, -1)
    
    # Apply preprocessing
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    confidence = max(model.predict_proba(input_scaled)[0])*100

    # Human-readable output
    if prediction == 1:
        result = f"Diabetic (Model Confidence: {confidence:.2f}%)"
    else:
        result = f"Non-Diabetic (Model Confidence: {confidence:.2f}%)"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
