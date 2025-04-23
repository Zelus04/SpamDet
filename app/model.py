# app/model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=["label", "message"])

# Preprocess
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train
pipeline.fit(df['message'], df['label'])

# Save model
joblib.dump(pipeline, 'app/spam_model.pkl')
print("✅ Spam detection model trained and saved.")
