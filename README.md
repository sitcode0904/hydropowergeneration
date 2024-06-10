# hydropowergeneration
 This provides a comprehensive solution for predicting hydropower generation using a CNN-LSTM model based on various environmental and operational features.The project includes a Jupyter notebook for data preprocessing, model training, and evaluation, as well as a Flask web application for making predictions.
 Here's a README file for your repository on hydropower generation prediction:

---
## Repository Contents

- `hydro.ipynb`: Jupyter notebook containing the data preprocessing, model training, and evaluation code.
- `app.py`: Flask web application for making predictions using the trained model.
- `scalerx.pkl`: Scaler used for normalizing the input features.
- `label_encoderx.pkl`: Label encoder used in the preprocessing.
- `hydropowercheck_model.h5`: Trained CNN-LSTM model for predicting hydropower generation.
- `templates/index.html`: HTML template for the web application's interface.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/hydropower-generation-prediction.git
    cd hydropower-generation-prediction
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure you have the necessary model and scaler files (`hydropowercheck_model.h5`, `scalerx.pkl`, `label_encoderx.pkl`) in the project directory.

## Usage

### Running the Jupyter Notebook

1. Open the Jupyter notebook:
    ```sh
    jupyter notebook hydro.ipynb
    ```

2. Follow the steps in the notebook to preprocess the data, train the CNN-LSTM model, and evaluate its performance.

### Running the Flask Application

1. Start the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Use the web interface to input the required features and get the predicted hydropower generation.

## Project Structure

### hydro.ipynb

This notebook includes:

- Data loading and preprocessing
- Feature scaling
- Model building (CNN-LSTM)
- Model training and evaluation
- Visualization of training history
- Saving the trained model and scalers

### app.py

This Flask web application includes:

- Loading the trained model and scalers
- Routes for rendering the homepage and handling predictions
- Handling input data from the user, making predictions, and displaying results

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Joblib](https://joblib.readthedocs.io/)
- [Jupyter](https://jupyter.org/)

---
