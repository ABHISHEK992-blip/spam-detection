import pandas as pd
from utils import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

# Load dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels → numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
df['message'] = df['message'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
preds = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", accuracy_score(y_test, preds))

# Save model + vectorizer
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

print("✅ Model trained and saved!")