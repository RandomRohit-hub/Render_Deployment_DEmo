from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model from the .pkl file
# This code runs when Gunicorn starts the server.
try:
    model = joblib.load("Finalmodel.pkl")
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    model = None
    print("Error: Model file 'Finalmodel.pkl' not found.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, makes a prediction, and returns the result."""
    if model is None:
        return render_template('index.html', prediction_text='Error: Model is not loaded.')

    try:
        # Extract features from the form in the correct order and data types.
        features = {
            'IQ': [int(request.form['IQ'])],
            'CGPA': [float(request.form['CGPA'])],
            'Academic_Performance': [int(request.form['Academic_Performance'])],
            'Internship_Experience': [request.form['Internship_Experience']], # This is a string ('Yes'/'No')
            'Communication_Skills': [int(request.form['Communication_Skills'])],
            'Projects_Completed': [int(request.form['Projects_Completed'])]
        }

        # Create a pandas DataFrame. The model pipeline expects this format.
        input_df = pd.DataFrame(features)

        # Make prediction using the loaded pipeline
        prediction = model.predict(input_df)

        # Format the output for display
        output = 'Placed' if prediction[0] == 'Yes' else 'Not Placed'

    except Exception as e:
        # Handle potential errors during prediction
        output = f"An error occurred: {e}"

    # Render the page again with the prediction result
    return render_template('index.html', prediction_text=f'Prediction: {output}')

# This block only runs when you execute `python app.py` directly
# It will NOT run on the Render server with Gunicorn
if __name__ == "__main__":
    print("✅ Flask server is running in debug mode locally!")
    app.run(debug=True)
