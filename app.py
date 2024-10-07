import joblib
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('customer_churn_model.pkl')  # Ensure this path is correct
scaler = joblib.load('scaler.pkl')  # Ensure this path is correct

# Load feature names from the file
with open('feature_names.txt', 'r') as f:
    expected_features = f.read().splitlines()

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Customer Churn Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
            form { display: grid; gap: 10px; }
            label { font-weight: bold; }
            input, select { width: 100%; padding: 5px; }
            input[type="submit"] { background-color: #4CAF50; color: white; border: none; padding: 10px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h2>Customer Churn Prediction</h2>
        <p>Enter customer information:</p>
        <form action="/predict" method="POST">
            <label for="age">Age:</label>
            <input type="number" id="age" name="Age" required min="18" max="120">

            <label for="tenure">Tenure (months):</label>
            <input type="number" id="tenure" name="Tenure" required min="0">

            <label for="usage">Monthly Usage (GB):</label>
            <input type="number" id="usage" name="MonthlyUsage" required min="0" step="0.1">

            <label for="contract">Contract Type:</label>
            <select id="contract" name="Contract">
                <option value="Month-to-Month">Month-to-Month</option>
                <option value="One Year">One Year</option>
                <option value="Two Year">Two Year</option>
            </select>

            <label for="internetType">Internet Type:</label>
            <select id="internetType" name="InternetType">
                <option value="Fiber Optic">Fiber Optic</option>
                <option value="Cable">Cable</option>
                <option value="DSL">DSL</option>
                <option value="No Internet">No Internet</option>
            </select>

            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create a dictionary with default values for all expected features
        data = {feature: 0 for feature in expected_features}

        # Update the dictionary with the values from the form
        form_data = {
            'Age': int(request.form['Age']),
            'Tenure': int(request.form['Tenure']),
            'MonthlyUsage': float(request.form['MonthlyUsage']),
            f"Contract_{request.form['Contract']}": 1,
            f"InternetType_{request.form['InternetType']}": 1  # Fixed naming convention here
        }
        data.update(form_data)

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Ensure all columns are in the correct order
        input_data = input_data[expected_features]

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)[0][1]  # Probability of churning

        # Interpret prediction result
        result = "Likely to Churn" if prediction[0] == 1 else "Not Likely to Churn"

        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Churn Prediction Result</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; text-align: center; }
                .result { font-size: 24px; margin: 20px 0; }
                .probability { font-size: 18px; margin-bottom: 20px; }
                .back-link { display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; }
            </style>
        </head>
        <body>
            <h2>Churn Prediction Result</h2>
            <div class="result">''' + result + '''</div>
            <div class="probability">Probability of churning: ''' + f"{probability:.2f}" + '''</div>
            <a href="/" class="back-link">Make Another Prediction</a>
        </body>
        </html>
        ''')

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
