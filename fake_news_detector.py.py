import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real])

# Select text and labels
X = data["text"]
y = data["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vector = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# Test with custom input
news = input("Enter a news headline: ")
news_vector = vectorizer.transform([news])
prediction = model.predict(news_vector)

if prediction[0] == 1:
    print("Real News")
else:
    print("Fake News")