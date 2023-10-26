from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get logs from the POST request
    data = request.get_json(force=True)
    log = data['log']

    # Vectorize the log
    log_vectorized = vectorizer.transform([log])

    # Make a prediction
    prediction = model.predict(log_vectorized)

    # Convert prediction to FAILURE or SUCCESS
    output = 'FAILURE' if prediction[0] == 1 else 'SUCCESS'

    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
