import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Sample Dataset (Ham/Spam)
# Using a small synthetic dataset for demonstration. 
# In a real scenario, we would load 'spam.csv'
data = {
    'message': [
        'Win a free lottery now', 'Call this number for free money', 'You have won a prize', 
        'Exclusive offer just for you', 'Credit card offer', 'Urgent! Claim your refund',
        'Meeting at 3pm', 'Hey, how are you?', 'Can we catch up tomorrow?',
        'Project update', 'Lunch plans?', 'Don\'t forget the documents',
        'Hello, this is a friendly reminder', 'Are you coming to the party?',
        'Limited time offer! Buy now!'
    ],
    'label': [
        'spam', 'spam', 'spam', 'spam', 'spam', 'spam',
        'ham', 'ham', 'ham', 'ham', 'ham', 'ham',
        'ham', 'ham', 'spam'
    ]
}

df = pd.DataFrame(data)

# Preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(X)

# Training
clf = MultinomialNB()
clf.fit(X, y)

# Saving Model and Vectorizer
model_path = 'model/spam_classifier.pkl'
vectorizer_path = 'model/vectorizer.pkl'

if not os.path.exists('model'):
    os.makedirs('model')

with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(cv, f)

print("Model and Vectorizer saved successfully!")
