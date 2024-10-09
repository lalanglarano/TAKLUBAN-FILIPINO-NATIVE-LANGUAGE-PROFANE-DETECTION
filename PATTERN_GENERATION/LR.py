import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Load the CSV file
df = pd.read_csv('UsedDataset/dataset_tagalog_sentence_profane.csv')

# Check class distribution
print("Class distribution in the dataset:")
print(df['profane'].value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['profane'], test_size=0.30, random_state=50)

# Check class distribution in the training and test sets
print("Class distribution in the training set:")
print(y_train.value_counts())
print("Class distribution in the test set:")
print(y_test.value_counts())

# Create a pipeline that combines the CountVectorizer and LogisticRegression with increased regularization
model = make_pipeline(CountVectorizer(), LogisticRegression(C=0.1))

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}')

# Train the model
model.fit(X_train, y_train)

# Function to predict if a sentence contains profane language
def predict_profane(sentence):
    return model.predict([sentence])

# Example usage
sentence = "tangina non ako ba naman sisihin"
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
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions)
    
    return accuracy, precision, recall, f1, cm, cr

# Evaluate the model
accuracy, precision, recall, f1, cm, cr = evaluate_model(model, X_test, y_test)
print(f"Model Evaluation Metrics on Test Set:\n"
      f"Accuracy: {accuracy:.2f}\n"
      f"Precision: {precision:.2f}\n"
      f"Recall: {recall:.2f}\n"
      f"F1 Score: {f1:.2f}")
print(f"Confusion Matrix:\n{cm}")
print(f"Classification Report:\n{cr}")

# Save the model to a file
joblib.dump(model, 'profane_detector_model.pkl')
