from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your dataset
file_path = 'D:/case_analysis/sherlock_holmes_cases.csv'
df = pd.read_csv(file_path)

# Preprocessing the data
X = df['description']  # Features (case descriptions)
y = df['status']  # Target variable (status of the case: Solved/Unsolved)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_tfidf, y_train)


def predict(request):
    if request.method == 'POST':
        new_case_description = request.POST.get('description')

        new_case_tfidf = vectorizer.transform([new_case_description])
        prediction = classifier.predict(new_case_tfidf)
        predicted_status = prediction[0]

        # Generate detailed case analysis
        analysis = generate_case_analysis(new_case_description)

        return JsonResponse({
            'status': predicted_status,
            'observations': analysis['Observations'],
            'leads': analysis['Leads'],
            'next_steps': analysis['Next Steps']
        })

    return render(request, 'analysis/index.html')


def generate_case_analysis(new_case_description):
    # Initialize leads and other related variables
    leads = []
    observations = []

    # Transform the new case description
    new_case_vector = vectorizer.transform([new_case_description])
    case_vectors = vectorizer.transform(df['description'])
    similarities = cosine_similarity(new_case_vector, case_vectors).flatten()

    # Get top matching cases
    top_indices = similarities.argsort()[-5:][::-1]
    similar_cases = df.iloc[top_indices]

    for i, case in similar_cases.iterrows():
        confidence_score = round(similarities[i] * 100, 2)
        if confidence_score > 0:
            common_keywords = set(new_case_description.split()).intersection(set(case['description'].split()))
            common_keywords_str = ', '.join(common_keywords) if common_keywords else "No significant keywords found"
            observations.append(
                f"Similar case: '{case['description']}' with status '{case['status']}'. Common keywords: {common_keywords_str}.")
            leads.append({
                "Lead": f"Derived from case '{case['description']}'",
                "Confidence_Score": confidence_score,
                "Suggested_Action": f"Investigate suspects or motives linked to similar patterns observed in case '{case['description']}'."
            })

    next_steps = [
        "Review CCTV footage from the scene and surrounding areas.",
        "Cross-check witness statements for inconsistencies.",
        "Use forensic analysis to identify unique traces left by the culprit.",
        "Investigate connections between suspects from previous similar cases."
    ]

    return {
        "Observations": observations,
        "Leads": leads,
        "Next Steps": next_steps
    }
