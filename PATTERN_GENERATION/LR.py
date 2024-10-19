import os
import pandas as pd
import csv
from nltk.tag.stanford import StanfordPOSTagger
from nltk.util import ngrams
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

class PatternGenerator:
    def __init__(self, csv_filename, model_filename, path_to_jar):
        self.rules = self.load_predefined_rules(csv_filename)
        self.tagger = StanfordPOSTagger(model_filename=model_filename, path_to_jar=path_to_jar)
    
    def load_predefined_rules(self, csv_filename):
        rules = []
        try:
            with open(csv_filename, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    rules.append({
                        'Rule Name': row['Rule Name'],
                        'POS Pattern': row['POS Pattern'].split(),
                        'Description': row['Description']
                    })
        except Exception as e:
            print(f"Error loading rules from {csv_filename}: {e}")
        return rules

    def generate_ngrams(self, pos_tagged_text, n):
        pos_tags = [item.split('|')[-1] for item in pos_tagged_text if '|' in item]
        ngrams_list = list(ngrams(pos_tags, n))
        return ngrams_list

    def apply_rules_to_ngrams(self, ngram_list):
        flagged_patterns = []
        matching_ngram_indices = []

        for idx, ngram in enumerate(ngram_list):
            for rule in self.rules:
                pattern = rule['POS Pattern']
                if len(ngram) == len(pattern):
                    match_percentage = sum(1 for ng, pt in zip(ngram, pattern) if ng == pt) / len(pattern)
                    if match_percentage >= 0.75:
                        flagged_patterns.append(f"{rule['Rule Name']} - {rule['Description']}")
                        matching_ngram_indices.append(idx)
        return flagged_patterns, matching_ngram_indices

    def detect_profane_patterns(self, pos_tagged_text):
        results, profane_ngram_indices = [], []

        for n in range(1, 11):
            ngrams_list = self.generate_ngrams(pos_tagged_text, n)
            detected_patterns, ngram_indices = self.apply_rules_to_ngrams(ngrams_list)
            if detected_patterns:
                results += detected_patterns
                profane_ngram_indices += ngram_indices

        return results if results else ["No profane patterns detected"], profane_ngram_indices

    def tag_sentence(self, sentence):
        tokens = sentence.split()
        tagged_sentence = self.tagger.tag(tokens)
        return [f"{word}|{tag}" for word, tag in tagged_sentence]

    def censor_sentence(self, pos_tagged_sentence, profane_indices):
        return ' '.join(
            '*****' if idx in profane_indices else word.split('|')[0] 
            for idx, word in enumerate(pos_tagged_sentence)
        )

def load_profane_dictionary(file_path):
    profane_dict = {}
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 2:
                    word, count = row
                    profane_dict[word] = int(count)
    except FileNotFoundError:
        print(f"{file_path} not found. Creating a new dictionary.")
    return profane_dict

def save_profane_dictionary(profane_dict, file_path):
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for word, count in profane_dict.items():
                writer.writerow([word, count])
    except Exception as e:
        print(f"Error saving profane dictionary: {e}")

def predict_and_censor(sentence, pattern_generator, best_model):
    is_profane = best_model.predict([sentence])[0]
    if not is_profane:
        return sentence

    pos_tagged_sentence = pattern_generator.tag_sentence(sentence)
    detected_patterns, profane_ngram_indices = pattern_generator.detect_profane_patterns(pos_tagged_sentence)
    if "No profane patterns detected" in detected_patterns:
        return sentence

    return pattern_generator.censor_sentence(pos_tagged_sentence, profane_ngram_indices)

def main():
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    predefined_rules_path = f"{base_path}/PATTERN_GENERATION/predefined_rules.csv"
    model_filename = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
    profane_dictionary_path = 'PATTERN_GENERATION/profane_dictionary.csv'

    pattern_generator = PatternGenerator(predefined_rules_path, model_filename, path_to_jar)

    sentence = "pakyu hindi nga kasi ako yun"
    pos_tagged_sentence = pattern_generator.tag_sentence(sentence)

    df = pd.read_csv('UsedDataset/dataset_tagalog_sentence_profane.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['profane'], test_size=0.5, random_state=48)

    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), SVC())
    param_grid = {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(classification_report(y_test, best_model.predict(X_test)))

    prediction = predict_and_censor(sentence, pattern_generator, best_model)
    print(f"Output: {prediction}")

if __name__ == "__main__":
    main()
