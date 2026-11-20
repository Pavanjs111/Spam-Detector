import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("1. Loading data...")
# Load the dataset (adjust encoding for Windows/Kaggle files)
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only the columns we need and rename them
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

print("2. Cleaning text...")
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()      # collapse spaces
    return text

df["clean_message"] = df["message"].apply(clean_text)

print("3. Splitting data...")
X = df["clean_message"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("4. Training the model...")
# Create a pipeline: TF-IDF (convert text to numbers) -> Naive Bayes (classifier)
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

print("5. Evaluating the model...")
y_pred = model.predict(X_test)
print("\n--- Evaluation Results ---")
print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))

print("6. Saving the model...")
joblib.dump(model, "spam_model.joblib")
print("Done! Model saved as 'spam_model.joblib'")