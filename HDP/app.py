# app.py

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('knn_model.pkl')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = request.form['sex']
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])

        # Create a pandas DataFrame from user input
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol]],
                                  columns=['age', 'sex', 'cp', 'trestbps', 'chol'])

        # Make a prediction using the model
        prediction = model.predict(input_data)

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
