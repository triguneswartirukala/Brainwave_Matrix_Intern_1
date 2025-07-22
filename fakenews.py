# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. Load the cleaned dataset
df = pd.read_csv("cleaned_fake_news.csv")

# 3. Basic preprocessing (optional)
df = df[df['language'] == 'english']  # Filter only English articles
df = df.dropna(subset=['text', 'type'])

# 4. Feature and label
X = df['text']
y = df['type']  # Make sure your labels are like 'real' or 'fake'

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 7. Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 8. Make predictions
y_pred = model.predict(X_test_tfidf)

# 9. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
