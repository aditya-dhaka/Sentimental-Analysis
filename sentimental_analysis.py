import pandas as pd
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import re

# Download stopwords
nltk.download('stopwords')

# Reading dataset
df = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Preview the dataset
print(df.head())

# Preprocessing the text reviews
corpus = []
for i in range(0, len(df)):
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['Review'][i])  # Remove non-alphabetic characters
    review = review.lower()  # Convert to lowercase
    review_words = review.split()  # Tokenize the review
    review_words = [word for word in review_words if word not in set(stopwords.words('english'))]  # Remove stopwords
    ps = PorterStemmer()
    review_words = [ps.stem(word) for word in review_words]  # Stem the words
    review = ' '.join(review_words)  # Join words back into a single string
    corpus.append(review)

# Feature extraction using CountVectorizer
cv = CountVectorizer(max_features=500)
x = cv.fit_transform(corpus).toarray()

# Target variable (sentiment labels)
y = df.iloc[:, -1].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=104)

# Naive Bayes Classifiers
clf1 = GaussianNB()
clf2 = MultinomialNB()
clf3 = BernoulliNB()

# Train the models
clf1.fit(x_train, y_train)
clf2.fit(x_train, y_train)
clf3.fit(x_train, y_train)

# Predictions
y_predG = clf1.predict(x_test)
y_predM = clf2.predict(x_test)
y_predB = clf3.predict(x_test)

# Accuracy scores in percentage
accuracy_G = accuracy_score(y_test, y_predG) * 100
accuracy_M = accuracy_score(y_test, y_predM) * 100
accuracy_B = accuracy_score(y_test, y_predB) * 100

# Print results as percentages
print(f"Gaussian Naive Bayes accuracy: {accuracy_G:.2f}%")
print(f"Multinomial Naive Bayes accuracy: {accuracy_M:.2f}%")
print(f"Bernoulli Naive Bayes accuracy: {accuracy_B:.2f}%")

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
print(f"Random Forest accuracy: {accuracy_rf:.2f}%")

# XGBoost Classifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb) * 100
print(f"XGBoost accuracy: {accuracy_xgb:.2f}%")

# Confusion matrix for the best model (Example: XGBoost)
print("\nConfusion Matrix for XGBoost:")
print(confusion_matrix(y_test, y_pred_xgb))

# Summary of the dataset
print("\nDataset Information:")
print(df.info())
print("\nColumns in the dataset:", df.columns)

# Find the best performing model based on accuracy
accuracies = {
    'Gaussian Naive Bayes': accuracy_G,
    'Multinomial Naive Bayes': accuracy_M,
    'Bernoulli Naive Bayes': accuracy_B,
    'Random Forest': accuracy_rf,
    'XGBoost': accuracy_xgb
}

best_model = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model]

print(f"\nThe best performing model is {best_model} with an accuracy of {best_accuracy:.2f}%")
