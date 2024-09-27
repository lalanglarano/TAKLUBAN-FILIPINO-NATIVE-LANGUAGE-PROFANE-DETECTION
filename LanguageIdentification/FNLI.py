import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import re
import csv

# Preprocessing class for basic text cleaning and sentence splitting
class TextPreprocessor:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/dataset/dataset_{language}.csv"
        self.output_dir = f"{results_folder}/preprocessed/"
        self.output_file = f"{self.output_dir}/preprocessed_{language}.csv"
        self.dictionary_dir = f"{base_path}/LanguageIdentification/Dictionary/"
        self.dictionary_file = f"{self.dictionary_dir}/{language}_dictionary.csv"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dictionary_dir, exist_ok=True)

    def preprocess_text(self, text):
        # Remove starting double quotes in the sentence
        text = text.lstrip('"')

        # Convert to lowercase
        text = text.lower()

        # Remove non-alphanumeric characters except spaces and commas
        text = ''.join(char if char.isalnum() or char in [' ', ','] else '' for char in text)

        # Remove digits
        text = ''.join(char if not char.isdigit() else '' for char in text)

        return text

    def split_into_sentences(self, text):
        # Split the text based on punctuation marks: periods, exclamation marks, question marks, colons, and commas
        chunks = re.split(r'[.!?:]', text)

        sentences = []
        for chunk in chunks:
            # Split the chunk based on commas and filter out segments with fewer than 2 words
            sub_sentences = re.split(r',', chunk)
            for sub in sub_sentences:
                words = sub.strip().split()
                if len(words) >= 2:  # Accept only if 2 or more consecutive words
                    sentences.append(sub.strip())

        return sentences

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
                    sentences = self.split_into_sentences(preprocessed_line)

                    for sentence in sentences:
                        writer.writerow([sentence])

                        # Update the word count dictionary
                        words = sentence.split()
                        for word in words:
                            if not word.isnumeric():
                                word_count[word] = word_count.get(word, 0) + 1

            if word_count:
                with open(self.dictionary_file, 'w', newline='', encoding='utf-8') as dict_file:
                    writer = csv.writer(dict_file)
                    writer.writerow(['word', 'frequency'])
                    for word, freq in sorted(word_count.items()):
                        writer.writerow([word, freq])
                print(f"Dictionary saved at {self.dictionary_file}")
            else:
                print(f"No words found after preprocessing for {self.dictionary_file}")

        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Class for handling language identification with noise word removal
class LanguageIdentification:
    def __init__(self, data, dictionary_dir):
        self.data = data
        self.vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
        self.model = MultinomialNB()
        self.dictionaries = {}
        self.dictionary_dir = dictionary_dir  # Add this line

        # Define noise words for each language
        self.noise_words = {
            'Tagalog': set(["na", "nang", "ng", "mga", "ang", "kung", "yan", "yun", "ayan", "sina", "sila",
                            "baka", "ano", "anong", "mag", "doon", "mo", "so", "po", "ko", "eme", "may", 
                            "luh", "ito", "ay", "ganon", "lang", "dito", "pang", "daw", "raw"]),
            'Bikol': set(["da", "ngani", "tabi", "ning", "kamo", "ini", "iyo", "sin", "hali", "bala", 
                          "aba", "alin", "baga", "ganiyan", "gaya", "ho", "ika", "kay", "mo", "naman", 
                          "wag", "naman", "yata", "ba"]),
            'Cebuano': set(["dayon", "gani", "kana", "mao", "diay", "mao ni", "mao ba", "lang", "usa", 
                            "kita", "kita tanan", "kamo", "ta", "gyud", "bitaw", "pud", "kay", "ahh", 
                            "pag", "pwede", "pwes", "pano", "ug"])
        }

    def load_dictionaries(self):
        for language in self.noise_words.keys():
            dict_file = f"{self.dictionary_dir}/{language.lower()}_dictionary.csv"
            if os.path.exists(dict_file):
                df = pd.read_csv(dict_file)
                self.dictionaries[language] = dict(zip(df['word'], df['frequency']))
            else:
                print(f"Warning: Dictionary file {dict_file} not found.")

    def remove_noise_words(self, sentences, label):
        noise_words_for_language = self.noise_words.get(label, set())
        cleaned_sentences = []
        
        for sentence in sentences:
            cleaned_sentence = ' '.join(word for word in sentence.split() if word not in noise_words_for_language)
            cleaned_sentences.append(cleaned_sentence)
        
        return cleaned_sentences

    def prepare_data(self):
        sentences = self.data['sentence'].values
        labels = self.data['label'].values
        
        # Remove noise words for each language
        cleaned_sentences = []
        for sentence, label in zip(sentences, labels):
            cleaned_sentences += self.remove_noise_words([sentence], label)
        
        X = self.vectorizer.fit_transform(cleaned_sentences).toarray()

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

# Main flow for preprocessing and language identification
if __name__ == "__main__":
    languages = ['tagalog', 'bikol', 'cebuano']

    # Preprocess files for Tagalog, Bikol, and Cebuano
    for language in languages:
        processor = TextPreprocessor(language)
        processor.preprocess_csv()

    # Load the preprocessed datasets
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    results_folder = f"{base_path}/Results"
    
    tagalog_output_file = f"{results_folder}/preprocessed/preprocessed_tagalog.csv"
    bikol_output_file = f"{results_folder}/preprocessed/preprocessed_bikol.csv"
    cebuano_output_file = f"{results_folder}/preprocessed/preprocessed_cebuano.csv"
    
    tagalog_data = pd.read_csv(tagalog_output_file, names=['sentence'])
    tagalog_data['label'] = 'Tagalog'
    
    bikol_data = pd.read_csv(bikol_output_file, names=['sentence'])
    bikol_data['label'] = 'Bikol'
    
    cebuano_data = pd.read_csv(cebuano_output_file, names=['sentence'])
    cebuano_data['label'] = 'Cebuano'
    
    combined_data = pd.concat([tagalog_data, bikol_data, cebuano_data], ignore_index=True)

    # Initialize LanguageIdentification and load dictionaries
    language_identifier = LanguageIdentification(data=combined_data, dictionary_dir=processor.dictionary_dir)
    language_identifier.load_dictionaries()

    # Prepare data and train model
    X_train, X_test, y_train, y_test = language_identifier.prepare_data()
    language_identifier.train_model(X_train, y_train)

    # Evaluate the model
    language_identifier.evaluate_model(X_test, y_test)

    # Test a sample sentence
    test_sentence = "Tayo ay magbasa putangina"
    predicted_language = language_identifier.predict_language(test_sentence)
    print(f"The predicted language for the sentence '{test_sentence}' is: {predicted_language}")
