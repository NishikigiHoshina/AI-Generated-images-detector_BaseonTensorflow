
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing.text_preprocessor import preprocess_text, vectorizer

# Load the dataset
data = pd.read_csv('data/news_data.csv')

# Preprocess the text
data['text'] = data['text'].fillna('').apply(preprocess_text)

# Convert text data into numerical features
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Print vocabulary size and label distribution for debugging
print("TF-IDF vocabulary size:", len(vectorizer.vocabulary_))
print("Label distribution:")
print(y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
