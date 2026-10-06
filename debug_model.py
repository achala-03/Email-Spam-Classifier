import pickle
import os

model_path = 'model/spam_classifier.pkl'
vectorizer_path = 'model/vectorizer.pkl'

print(f"Loading from {os.getcwd()}")

try:
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    print("Model loaded.")

    with open(vectorizer_path, 'rb') as f:
        cv = pickle.load(f)
    print("Vectorizer loaded.")

    message = "Win a free lottery now"
    vect = cv.transform([message]).toarray()
    print(f"Vector shape: {vect.shape}")
    
    prediction = clf.predict(vect)
    print(f"Prediction: {prediction}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
