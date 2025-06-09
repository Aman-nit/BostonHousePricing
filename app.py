import pickle 
from flask import Flask,request,app,jsonify,render_template,url_for
import numpy as np 
import pandas as pd 

app = Flask(__name__)
##Load my model file 
remodel = pickle.load(open('reModel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Ensure request contains JSON and 'data'
    if not request.json or 'data' not in request.json:
        return jsonify({'error': 'Missing "data" key in request'}), 400

    data = request.json['data']

    # Ensure the data is a 2D list
    if not isinstance(data, list) or not isinstance(data[0], list):
        return jsonify({'error': 'Data must be a 2D array like [[...]]'}), 400

    # Define the column names
    column_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    # Create DataFrame
    input_df = pd.DataFrame(data, columns=column_names)

    # Scale input
    new_data = scaler.transform(input_df)

    # Predic
    output = remodel.predict(new_data)

    # Return prediction
    return jsonify({'prediction': output.tolist()})



@app.route('/Predict', methods=['POST'])
def predict():
    form_data = request.form.values()
    float_data = [float(x) for x in form_data] 

    # Convert to 2D array
    data = [float_data]  # shape (1, 13)

    # Define the column names
    column_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    # Create DataFrame
    input_df = pd.DataFrame(data, columns=column_names)

    # Scale input
    new_data = scaler.transform(input_df)

    # Predict
    output = remodel.predict(new_data)

    return f"Predicted House Price: ${output[0]:.2f}"


if __name__ == '__main__':
    app.run(debug=True)