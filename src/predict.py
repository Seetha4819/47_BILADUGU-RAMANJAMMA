import pickle
import re

# Load trained model
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

message = input("Enter SMS message: ")

message_clean = clean_text(message)
message_vec = vectorizer.transform([message_clean])

prediction = model.predict(message_vec)[0]

if prediction == 1:
    print("🚨 Spam Message Detected")
else:
    print("✅ This is a Legitimate Message (Ham)")
