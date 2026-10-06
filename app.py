from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load Model and Vectorizer
model_path = 'model/spam_classifier.pkl'
vectorizer_path = 'model/vectorizer.pkl'

# Ensure model files exist
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    print("Error: Model or Vectorizer file not found. Please run train_model.py first.")
    exit(1)

with open(model_path, 'rb') as f:
    clf = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    cv = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request")
        data = request.get_json(force=True, silent=True)
        print(f"Data: {data}")
        
        if not data:
            print("No JSON data found")
            return jsonify({'error': 'No JSON data provided or invalid JSON'}), 400
            
        message = data.get('message', '')
        print(f"Message: {message}")
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Preprocess and Vectorize
        vect = cv.transform([message]).toarray()
        
        # Predict
        prediction = clf.predict(vect)
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
        print(f"Prediction: {result}")
        
        return jsonify({'prediction': result})
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Host='0.0.0.0' allows access from network
    app.run(debug=True, host='0.0.0.0', port=5001)
