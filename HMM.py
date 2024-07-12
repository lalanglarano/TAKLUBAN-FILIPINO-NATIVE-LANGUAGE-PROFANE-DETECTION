import os
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import csv
import re
import string

class TextPreprocessor:
    def __init__(self, language):
        base_path = 'C:/Users/ADMIN/OneDrive/ドキュメント/Annalyn/THESIS/TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/'
        self.input_file = f"{base_path}/dataset/dataset_{language}.csv"
        self.output_dir = f"{base_path}/preprocessed/"
        self.output_file = f"{self.output_dir}/preprocessed_{language}.csv"
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def preprocess_text(text):
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

    # HHM Start------------------------------------------- #

    # Prepare the data
    sentences = data['sentence'].values
    labels = data['label'].values
    # Convert the text data into a matrix of token counts. It uses character n-grams of length 1 and 2 (unigrams and bigrams).
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences).toarray()

    # Encode labels
    # dictionaries for mapping between labels and indices.
    label_to_index = {label: idx for idx, label in enumerate(set(labels))}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y = np.array([label_to_index[label] for label in labels])

    # Split the data into training and testing sets 70-30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train separate HMMs for each language
    hmm_models = {}
    for label, idx in label_to_index.items():
        # Ensure enough samples to fit the model
        if np.sum(y_train == idx) >= 3:  # Minimum number of samples should be >= number of clusters
            model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)
            model.fit(X_train[y_train == idx])
            hmm_models[label] = model
        else:
            print(f"Not enough samples to train HMM for {label} language.")

    # Predict the language of new sentences
    # takes a sentence, vectorizes it, and scores it against all HMM models.
    def predict_language(sentence):
        X_new = vectorizer.transform([sentence]).toarray()
        scores = {label: model.score(X_new) for label, model in hmm_models.items()}
        return max(scores, key=scores.get)

    # Predictions stored in y_pred
    y_pred = []
    for sentence in X_test:
        sentence_text = ' '.join(vectorizer.inverse_transform(sentence.reshape(1, -1))[0])
        predicted_label = predict_language(sentence_text)
        y_pred.append(label_to_index[predicted_label])

    y_pred = np.array(y_pred)

    # Calculate Performance Metrics precision, recall, F1-score, and accuracy
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    # Example prediction with user input
    # while True:
    #    new_sentence = input("Enter a sentence to predict its language (or type 'exit' to quit): ")
    #   if new_sentence.lower() == 'exit':
    #        break
    #    new_sentence = TextPreprocessor.preprocess_text(new_sentence)
    #    predicted_language = predict_language(new_sentence)
    #    print(f"The predicted language for '{new_sentence}' is {predicted_language}.")
    #    try_again = input("Do you want to try again? (yes/no): ").strip().lower()
    #    if try_again != 'yes':
    #        break

