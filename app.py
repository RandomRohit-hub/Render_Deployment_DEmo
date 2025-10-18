from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = joblib.load("Finalmodel.pkl")

@app.route('/')
def home():
    return "✅ ML Model API is Running!"

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from user
        data = request.get_json()

        # Convert JSON to DataFrame (since model expects tabular input)
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)

        # Return result
        return jsonify({
            "prediction": prediction.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)





