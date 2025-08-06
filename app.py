from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.lb')

# Define categorical options for encoding
encoders = {}
category_maps = {
    'job': ['unemployed', 'services', 'management', 'blue-collar',
       'self-employed', 'technician', 'entrepreneur', 'admin.', 'student',
       'housemaid', 'retired', 'unknown'],
    'marital': ['divorced', 'married', 'single'],
    'education': ['primary', 'secondary', 'tertiary', 'unknown'],
    'housing': ['yes', 'no']
}

# Create label encoders for each category
for col, classes in category_maps.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes) # 👈 FIXED: convert to numpy array
    encoders[col] = le

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/project', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        age = int(request.form['age'])
        job = request.form['job']
        marital = request.form['marital']
        education = request.form['education']
        balance = int(request.form['balance'])
        housing = request.form['housing']
        duration = int(request.form['duration'])
        campaign = int(request.form['campaign'])

        # Encode categorical values
        job_encoded = encoders['job'].transform([job])[0]
        marital_encoded = encoders['marital'].transform([marital])[0]
        education_encoded = encoders['education'].transform([education])[0]
        housing_encoded = encoders['housing'].transform([housing])[0]

        # Prepare input for prediction
        input_data = [
            age,
            job_encoded,
            marital_encoded,
            education_encoded,
            balance,
            housing_encoded,
            duration,
            campaign
        ]

        pred = model.predict([input_data])[0]
        prediction = "Yes" if pred == 1 else "No"

    return render_template('project.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
