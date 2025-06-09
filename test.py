import pickle
import numpy as np

# Load
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Sample input
sample = {
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.09,
    "RAD": 1,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.90,
    "LSTAT": 4.98
}

try:
    input_array = np.array(list(sample.values())).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    print("✅ Working fine! Prediction:", prediction[0])
except Exception as e:
    print("❌ Error found:", str(e))
