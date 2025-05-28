# 📌 1️⃣ IMPORT LIBRARIES
import pandas as pd
import numpy as np
import re
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from textblob import TextBlob

import warnings
warnings.filterwarnings('ignore')

# 📌 2️⃣ DOWNLOAD NLTK RESOURCES (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# 📌 3️⃣ LOAD DATA
file_path = 'ai_dev_assignment_tickets_complex_1000.xls'
df = pd.read_excel(file_path, engine='xlrd')

# 📌 3.1 HANDLE MISSING DATA
df['ticket_text'] = df['ticket_text'].fillna("")
df['issue_type'] = df['issue_type'].fillna("Unknown")
df['urgency_level'] = df['urgency_level'].fillna("Unknown")
df['product'] = df['product'].fillna("Unknown")

# Check for columns
assert 'ticket_text' in df.columns, "'ticket_text' column not found in data"
assert 'issue_type' in df.columns, "'issue_type' column not found in data"
assert 'urgency_level' in df.columns, "'urgency_level' column not found in data"
assert 'product' in df.columns, "'product' column not found in data"

# 📌 4️⃣ CLEAN & PREPROCESS TEXT
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df['clean_text'] = df['ticket_text'].apply(preprocess_text)

# 📌 5️⃣ FEATURE ENGINEERING
df['ticket_length'] = df['ticket_text'].apply(lambda x: len(str(x).split()))
df['sentiment'] = df['ticket_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# 📌 6️⃣ SPLIT DATA
X = df['clean_text']
y_issue = df['issue_type']
y_urgency = df['urgency_level']

# Use the same split for both targets to maintain alignment
X_train, X_test, y_train_issue, y_test_issue, y_train_urgency, y_test_urgency = train_test_split(
    X, y_issue, y_urgency, test_size=0.2, random_state=42)

# 📌 7️⃣ TF-IDF VECTORIZE TEXT
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 📌 8️⃣ ISSUE TYPE CLASSIFIER (Random Forest)
issue_model = RandomForestClassifier(random_state=42)
issue_model.fit(X_train_vec, y_train_issue)
issue_pred = issue_model.predict(X_test_vec)
print("📊 Issue Type Classification Report:")
print(classification_report(y_test_issue, issue_pred))

# 📌 9️⃣ URGENCY LEVEL CLASSIFIER (Logistic Regression)
urgency_model = LogisticRegression(max_iter=500)
urgency_model.fit(X_train_vec, y_train_urgency)
urgency_pred = urgency_model.predict(X_test_vec)
print("📊 Urgency Level Classification Report:")
print(classification_report(y_test_urgency, urgency_pred))

# 📌 🔟 ENTITY EXTRACTION FUNCTION
def extract_entities(text, product_list, keywords):
    text_lower = text.lower()
    entities = {}
    entities['products'] = [product for product in product_list if product.lower() in text_lower]
    # Dates: handles formats like dd/mm/yyyy or yyyy-mm-dd
    entities['dates'] = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', text)
    entities['complaints'] = [word for word in keywords if word in text_lower]
    return entities

# 📌 1️⃣1️⃣ PREDICTION + EXTRACTION FUNCTION
def predict_ticket(ticket_text):
    processed_text = preprocess_text(ticket_text)
    text_vector = vectorizer.transform([processed_text])

    issue_type = issue_model.predict(text_vector)[0]
    urgency_level = urgency_model.predict(text_vector)[0]

    product_list = df['product'].dropna().unique().tolist()
    complaint_keywords = ['broken', 'late', 'error', 'failed', 'cancelled', 'slow']

    entities = extract_entities(ticket_text, product_list, complaint_keywords)

    result = {
        'predicted_issue_type': issue_type,
        'predicted_urgency_level': urgency_level,
        'extracted_entities': entities
    }
    return result

# 📌 1️⃣2️⃣ TEST SAMPLE TICKET
test_ticket = "I received a broken laptop on 22/05/2025 and it's very slow."
result = predict_ticket(test_ticket)
print("📑 Prediction Result:")
print(json.dumps(result, indent=4))