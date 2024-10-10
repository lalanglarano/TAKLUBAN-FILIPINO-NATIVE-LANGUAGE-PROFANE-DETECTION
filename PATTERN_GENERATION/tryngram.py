import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('UsedDataset/dataset_tagalog_sentence_profane.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['profane'], test_size=0.2, random_state=48)

# Create a pipeline that combines the TfidfVectorizer with N-Grams and SVM to reduce overfitting
pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), SVC())

# Define the hyperparameters grid
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf']
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Function to predict if a sentence contains profane language
def predict_profane(sentence):
    return best_model.predict([sentence])

# Example usage
sentence = "nakaka putangina dito sa bahay"
prediction = predict_profane(sentence)
print(f"The sentence '{sentence}' is classified as {'profane' if prediction == 1 else 'not profane'}.")

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

# Evaluate the model
accuracy, precision, recall, f1= evaluate_model(best_model, X_test, y_test)
print(f"Model Evaluation Metrics on Test Set:\n"
      f"Accuracy: {accuracy:.2f}\n"
      f"Precision: {precision:.2f}\n"
      f"Recall: {recall:.2f}\n"
      f"F1 Score: {f1:.2f}")

# Save the best model to a file
joblib.dump(best_model, 'ngram_svm_profane_detector_model.pkl')

# Plotting the distribution of n-grams in the dataset

# Create a TfidfVectorizer with n-gram range (1, 2)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['sentence'])

# Get feature names (n-grams)
feature_names = vectorizer.get_feature_names_out()

# Sum the tf-idf values for each n-gram across all documents
tfidf_sum = X.sum(axis=0).A1

# Create a DataFrame with n-grams and their corresponding tf-idf sums
ngram_tfidf = pd.DataFrame({'ngram': feature_names, 'tfidf': tfidf_sum})

# Sort the DataFrame by tf-idf values in descending order
ngram_tfidf = ngram_tfidf.sort_values(by='tfidf', ascending=False)

# Plot the distribution of the top 20 n-grams
top_20_ngrams = ngram_tfidf.head(20)
plt.figure(figsize=(10, 6))
plt.barh(top_20_ngrams['ngram'], top_20_ngrams['tfidf'], color='skyblue')
plt.xlabel('TF-IDF')
plt.ylabel('N-gram')
plt.title('Top 20 N-grams by TF-IDF')
plt.gca().invert_yaxis()
plt.show()
