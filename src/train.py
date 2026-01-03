import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# 2. Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3. Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data['message'] = data['message'].apply(clean_text)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 5. Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Test model
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Save model
with open("models/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully")
