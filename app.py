from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('xgboost_model.pkl')

# Define feature names
feature_names = [
    'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'PhysActivity', 
    'Fruits', 'Veggies', 'AnyHealthcare', 'GenHlth', 
    'MentHlth', 'PhysHlth', 'Sex', 'Age', 'Education', 'Income'
]

# Render the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        features = [float(request.form[feature]) for feature in feature_names]

        # Create a DataFrame with the input values
        input_data = pd.DataFrame([features], columns=feature_names)

        # Make predictions using the trained model
        prediction = model.predict(input_data)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
