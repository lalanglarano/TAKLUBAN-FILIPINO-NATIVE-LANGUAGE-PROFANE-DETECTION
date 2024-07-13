import os
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import csv
import re
import string

class TextPreprocessor:
    def __init__(self, language):
        base_path = 'C:/Users/ADMIN/OneDrive/ドキュメント/Annalyn/THESIS/TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/'
        self.input_file = f"{base_path}/dataset/dataset_{language}.csv"
        self.output_dir = f"{base_path}/preprocessed/"
        self.output_file = f"{self.output_dir}/preprocessed_{language}.csv"
        self.noise_words = set(["na", "nang", "ng", "mga", "ang", "kung", "yan", 
                                "yun", "ayan", "sina", "sila", "baka", "ano", "anong",
                                "mag", "doon", "si", "siya", "mo", "so", "ako", "ikaw", "po", "ko",
                                "eme", "may", "luh", "ito", "ay", "ganon", "basta", "lang", "dito",
                                "and", "i", "haha", "o", "pang", "daw", "raw", "aww", "kahit", "go",
                                "rin", "din", "kayo", "baka", "hoy", "ok", "okay", "yung", "yay",
                                "sa", "sabi", "eh", "sana"])
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def preprocess_text(self, text):
        # Convert text to lowercase
        text = text.lower()
        # Handle words with apostrophes
        text = re.sub(r"(\w)\'(\w)", r"\1\2", text)
        # Remove punctuation and special characters except numbers
        text = ''.join(char for char in text if char.isalnum() or char == ' ')
        # Remove isolated numbers (numbers surrounded by whitespace)
        text = re.sub(r'\b\d+\b', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove noise words
        text = ' '.join(word for word in text.split() if word not in self.noise_words)
        return text

    def preprocess_csv(self):
        try:
            if not os.path.exists(self.input_file):
                print(f"Error: The file {self.input_file} does not exist.")
                return
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                # Read the entire file as lines (treating each line as a sentence)
                lines = infile.readlines()
                writer = csv.writer(outfile)

                for line in lines:
                    # Process each line (assuming each line is a sentence)
                    preprocessed_line = self.preprocess_text(line)
                    writer.writerow([preprocessed_line.strip()])  # Write each preprocessed line as a single row

        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Preprocess files for Tagalog, Bikol, and Cebuano
languages = ['tagalog', 'bikol', 'cebuano']

for language in languages:
    processor = TextPreprocessor(language)
    processor.preprocess_csv()

# Check if the preprocessed files exist before proceeding
base_path = 'C:/Users/ADMIN/OneDrive/ドキュメント/Annalyn/THESIS/TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/'
tagalog_output_file = f"{base_path}/preprocessed/preprocessed_tagalog.csv"
bikol_output_file = f"{base_path}/preprocessed/preprocessed_bikol.csv"
cebuano_output_file = f"{base_path}/preprocessed/preprocessed_cebuano.csv"

if not all(os.path.exists(file) for file in [tagalog_output_file, bikol_output_file, cebuano_output_file]):
    print("Preprocessing failed or input files are missing. Exiting.")
else:
    # Load the datasets
    tagalog_data = pd.read_csv(tagalog_output_file, names=['sentence'])
    bikol_data = pd.read_csv(bikol_output_file, names=['sentence'])
    cebuano_data = pd.read_csv(cebuano_output_file, names=['sentence'])

    # Combine datasets and label them
    tagalog_data['label'] = 'Tagalog'
    bikol_data['label'] = 'Bikol'
    cebuano_data['label'] = 'Cebuano'

    data = pd.concat([tagalog_data, bikol_data, cebuano_data])

    # Remove rows with NaN values
    data = data.dropna()

class LanguageIdentification:  # HMM starts here
    def __init__(self, data):
        self.data = data
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
        self.hmm_models = {}

    def prepare_data(self):
        # Prepare the data
        sentences = self.data['sentence'].values
        labels = self.data['label'].values
        X = self.vectorizer.fit_transform(sentences).toarray()

        # Encode labels
        self.label_to_index = {label: idx for idx, label in enumerate(set(labels))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        y = np.array([self.label_to_index[label] for label in labels])

        return train_test_split(X, y, test_size=0.3, random_state=50)

    def train_models(self, X_train, y_train):
        for label, idx in self.label_to_index.items():
            # Ensure enough samples to fit the model
            if np.sum(y_train == idx) >= 3:  # Minimum number of samples should be >= number of clusters
                model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)
                model.fit(X_train[y_train == idx])
                self.hmm_models[label] = model
            else:
                print(f"Not enough samples to train HMM for {label} language.")
    
    def predict_language(self, sentence):
        X_new = self.vectorizer.transform([sentence]).toarray()
        scores = {label: model.score(X_new) for label, model in self.hmm_models.items()}
        return max(scores, key=scores.get)

    def evaluate_model(self, X_test, y_test):
        y_pred = []
        for sentence in X_test:
            sentence_text = ' '.join(self.vectorizer.inverse_transform(sentence.reshape(1, -1))[0])
            predicted_label = self.predict_language(sentence_text)
            y_pred.append(self.label_to_index[predicted_label])

        y_pred = np.array(y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate TP, TN, FP, FN
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)

        # Print metrics
        print(f"True Positives: {TP}")
        print(f"True Negatives: {TN}")
        print(f"False Positives: {FP}")
        print(f"False Negatives: {FN}")

        # Measurement of the Evaluation Metrics
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
lang_id.train_models(X_train, y_train)
lang_id.evaluate_model(X_test, y_test)

# Example prediction with user input
# while True:
#    new_sentence = input("Enter a sentence to predict its language (or type 'exit' to quit): ")
#    if new_sentence.lower() == 'exit':
#        break
#    new_sentence = TextPreprocessor.preprocess_text(new_sentence)
#    predicted_language = lang_id.predict_language(new_sentence)
#    print(f"The predicted language for '{new_sentence}' is {predicted_language}.")
#    try_again = input("Do you want to try again? (yes/no): ").strip().lower()
#    if try_again != 'yes':
#        break

