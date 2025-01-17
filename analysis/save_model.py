import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Sample data for training the model
data = {
    'description': [
        'Case 1 description',
        'Case 2 description',
        'Case 3 description',
        'Case 4 description'
    ],
    'status': ['Solved', 'Unsolved', 'Solved', 'Unsolved']
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df['description']
y = df['status']

# Create a TF-IDF vectorizer and transform the descriptions
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_tfidf, y)

# Save the trained model and vectorizer to files
joblib.dump(classifier, "random_forest_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
