import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Load the scaler and label encoder
scaler = joblib.load('scalerx.pkl')
label_encoder = joblib.load('label_encoderx.pkl')

# Load the model
model = load_model('hydropowercheck_model.h5', compile=False)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        precipitation = float(request.form['precipitation'])
        temperature = float(request.form['temperature'])
        reservoir_level = float(request.form['reservoir_level'])
        turbine_efficiency = float(request.form['turbine_efficiency'])
        flow_rate = float(request.form['flow_rate'])

        # Prepare the input data
        input_data = np.array([[precipitation, temperature, reservoir_level, turbine_efficiency, flow_rate]])
        input_data_scaled = scaler.transform(input_data)

        # Prepare the sequence for the model
        input_sequence = np.zeros((1, 5, 5))  # (batch_size, sequence_length, num_features)
        input_sequence[0, -1, :] = input_data_scaled  # Add the input data to the last time step

        # Make the prediction
        prediction_scaled = model.predict(input_sequence)

        # Since prediction_scaled is a NumPy array, we need to get the value inside it
        prediction_value = prediction_scaled[0][0]

        return render_template('index.html', prediction_text='Predicted Hydropower Generation: {:.2f} (MU)'.format(prediction_value))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)





