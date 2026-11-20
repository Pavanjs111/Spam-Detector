import joblib
import re

# 1. Load the brain we just saved
print("Loading AI model...")
model = joblib.load("spam_model.joblib")

# 2. We must use the EXACT SAME cleaning function we used in training!
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 3. Create a loop so you can keep typing messages
print("\n--- AI Spam Detector ---")
while True:
    message = input("\nEnter a message (or type 'quit' to stop): ")
    
    if message.lower() == 'quit':
        break
        
    # 4. Clean the message and ask the AI
    cleaned = clean_text(message)
    prediction = model.predict([cleaned])[0]
    
    # 5. Print the result
    if prediction == 'spam':
        print("🚨 Result: SPAM! Do not trust this message.")
    else:
        print("✅ Result: HAM (Normal Message).")