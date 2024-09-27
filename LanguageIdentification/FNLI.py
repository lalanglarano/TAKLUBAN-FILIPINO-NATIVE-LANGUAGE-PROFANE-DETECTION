import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from preProcess import TextPreprocessor  # Import the preprocessing class

languages = ['tagalog', 'bikol', 'cebuano']

for language in languages:
    processor = TextPreprocessor(language)
    processor.preprocess_csv()

# Load the datasets if the preprocessed files exist
base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
results_folder = f"{base_path}/Results"
tagalog_output_file = f"{results_folder}/preprocessed/preprocessed_tagalog.csv"
bikol_output_file = f"{results_folder}/preprocessed/preprocessed_bikol.csv"
cebuano_output_file = f"{results_folder}/preprocessed/preprocessed_cebuano.csv"

if not all(os.path.exists(file) for file in [tagalog_output_file, bikol_output_file, cebuano_output_file]):
    print("Preprocessing failed or input files are missing. Exiting.")
else:
    tagalog_data = pd.read_csv(tagalog_output_file, names=['sentence'])
    bikol_data = pd.read_csv(bikol_output_file, names=['sentence'])
    cebuano_data = pd.read_csv(cebuano_output_file, names=['sentence'])

    tagalog_data['label'] = 'Tagalog'
    bikol_data['label'] = 'Bikol'
    cebuano_data['label'] = 'Cebuano'

    data = pd.concat([tagalog_data, bikol_data, cebuano_data])
    data = data.dropna()

class LanguageIdentification:
    def __init__(self, data):
        self.data = data
        self.vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
        self.model = MultinomialNB()

    def prepare_data(self):
        sentences = self.data['sentence'].values
        labels = self.data['label'].values
        X = self.vectorizer.fit_transform(sentences).toarray()

        self.label_to_index = {label: idx for idx, label in enumerate(set(labels))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        y = np.array([self.label_to_index[label] for label in labels])

        return train_test_split(X, y, test_size=0.3, random_state=50)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict_language(self, sentence):
        X_new = self.vectorizer.transform([sentence]).toarray()
        predicted_idx = self.model.predict(X_new)[0]
        return self.index_to_label[predicted_idx]

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)

        print(f"True Positives: {TP}")
        print(f"True Negatives: {TN}")
        print(f"False Positives: {FP}")
        print(f"False Negatives: {FN}")

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

# Initialize and run the LanguageIdentification process
lang_id = LanguageIdentification(data)
X_train, X_test, y_train, y_test = lang_id.prepare_data()
lang_id.train_model(X_train, y_train)
lang_id.evaluate_model(X_test, y_test)