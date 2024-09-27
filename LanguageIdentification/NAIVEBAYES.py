import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import csv
import re
import string

class TextPreprocessor:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/dataset/dataset_{language}.csv"
        self.output_dir = f"{results_folder}/preprocessed/"
        self.output_file = f"{self.output_dir}/preprocessed_{language}.csv"
        self.dictionary_dir = f"{base_path}/LanguageIdentification/Dictionary/"  # Correct dictionary path
        self.dictionary_file = f"{self.dictionary_dir}/{language}_dictionary.csv"  # Ensure the dictionary file is saved as [language]_dictionary.csv
        self.noise_words = set(["na", "nang", "ng", "mga", "ang", "kung", "yan", "yun", "ayan", "sina", "sila",
                                "baka", "ano", "anong", "mag", "doon", "si", "siya", "mo", "so", "ako", "ikaw",
                                "po", "ko", "eme", "may", "luh", "ito", "ay", "ganon", "basta", "lang", "dito",
                                "and", "i", "haha", "o", "pang", "daw", "raw", "aww", "kahit", "go", "rin", "din",
                                "kayo", "baka", "hoy", "ok", "okay", "yung", "yay", "sa", "sabi", "eh", "sana",
                                "da", "ngani", "tabi", "ning", "kamo", "ini", "iyo", "sin", "kaya", "basta",
                                "hali", "bala", "aba", "alin", "baka", "baga", "ganiyan", "gaya", "ho", "ika",
                                "kay", "kumusta", "mo", "naman", "po", "sapagkat", "tayo", "talaga", "wag",
                                "naman", "yata", "ba", "bitaw", "dayon", "gani", "kana", "mao", "diay", "mao ni",
                                "mao ba", "lang", "usa", "kita", "kita tanan", "kamo", "ta", "lagi", "gyud",
                                "bitaw", "pud", "kay", "ahh", "pag", "pwede", "pwes", "pano", "ok", "ug"])
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dictionary_dir, exist_ok=True)

    def preprocess_text(self, text):
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = ''.join(char if char.isalnum() or char == ' ' else '' for char in text)
        # Remove noise words
        text = ' '.join(word for word in text.split() if word not in self.noise_words)
        return text

    def preprocess_csv(self):
        try:
            if not os.path.exists(self.input_file):
                print(f"Error: The file {self.input_file} does not exist.")
                return
            word_count = {}
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                lines = infile.readlines()
                writer = csv.writer(outfile)

                for line in lines:
                    preprocessed_line = self.preprocess_text(line)
                    writer.writerow([preprocessed_line.strip()])
                    
                    # Tokenize by word and update word_count dictionary
                    for word in preprocessed_line.split():
                        word_count[word] = word_count.get(word, 0) + 1

            # Save the word dictionary to a CSV file for each language in alphabetical order
            if word_count:  # Ensure the dictionary isn't empty
                with open(self.dictionary_file, 'w', newline='', encoding='utf-8') as dict_file:
                    writer = csv.writer(dict_file)
                    writer.writerow(['word', 'frequency'])
                    for word, freq in sorted(word_count.items()):  # Sort by word (alphabetically)
                        writer.writerow([word, freq])
                print(f"Dictionary saved at {self.dictionary_file}")
            else:
                print(f"No words found after preprocessing for {self.dictionary_file}")

        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Preprocess files for Tagalog, Bikol, and Cebuano and create dictionaries
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

class LanguageIdentification:
    def __init__(self, data):
        self.data = data
        self.vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')  # Use word-level tokenization
        self.model = MultinomialNB()

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
lang_id.train_model(X_train, y_train)
lang_id.evaluate_model(X_test, y_test)
