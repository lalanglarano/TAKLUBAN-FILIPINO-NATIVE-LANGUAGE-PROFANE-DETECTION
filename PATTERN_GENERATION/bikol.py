import os
import pandas as pd
import csv
import joblib
from nltk.tag.stanford import StanfordPOSTagger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from nltk.util import ngrams

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
        pos_tags = [item.split('|')[-1] for item in pos_tagged_text if '|' in item]  # Extract POS tags only
        print(f"Extracted POS Tags: {pos_tags}")
        ngrams_list = list(ngrams(pos_tags, n))
        print(f"Generated {n}-grams: {ngrams_list}")
        return ngrams_list

    def apply_rules_to_ngrams(self, ngram_list):
        flagged_patterns = []
        matching_ngram_indices = []
        for idx, ngram in enumerate(ngram_list):
            print(f"Checking n-gram: {ngram}")
            for rule in self.rules:
                pattern = rule['POS Pattern']
                if len(ngram) == len(pattern) and list(ngram) == pattern:
                    flagged_patterns.append(f"Rule Matched: {rule['Rule Name']} - {rule['Description']}")
                    matching_ngram_indices.append(idx)
                    print(f"Match found: {rule['Rule Name']}")
        return flagged_patterns, matching_ngram_indices

    def detect_profane_patterns(self, pos_tagged_text):
        results = []
        profane_ngram_indices = []
        for n in range(1, 11):  
            ngrams_list = self.generate_ngrams(pos_tagged_text, n)
            detected_patterns, ngram_indices = self.apply_rules_to_ngrams(ngrams_list)
            if detected_patterns:
                results += detected_patterns
                profane_ngram_indices += ngram_indices
        return results if results else ["No profane patterns detected"], profane_ngram_indices

    def save_patterns_from_sentence(self, csv_filename, sentence, description):
        pos_tagged_sentence = self.tag_sentence(sentence)
        print(f"POS-tagged Sentence: {pos_tagged_sentence}")
        try:
            pos_pattern = ' '.join([item.split('|')[-1] for item in pos_tagged_sentence if '|' in item])
            print(f"Extracted POS Pattern: {pos_pattern}")
        except IndexError:
            print("Error: Incorrect tagging format in sentence.")
            return
        rule_name = f"rule_from_sentence_{len(self.rules) + 1}"
        self.add_new_rule(csv_filename, rule_name, pos_pattern, description)
        print(f"New rule '{rule_name}' added with POS pattern: {pos_pattern}")

    def tag_sentence(self, sentence):
        tokens = sentence.split()
        tagged_sentence = self.tagger.tag(tokens)
        return [f"{word}|{tag}" for word, tag in tagged_sentence]

    def add_new_rule(self, csv_filename, rule_name, pos_pattern, description):
        current_rules = self.load_predefined_rules(csv_filename)
        for rule in current_rules:
            if rule['POS Pattern'] == pos_pattern.split():
                print(f"Rule '{rule_name}' with POS pattern '{pos_pattern}' already exists.")
                return
        new_rule = {'Rule Name': rule_name, 'POS Pattern': pos_pattern, 'Description': description}
        try:
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['Rule Name', 'POS Pattern', 'Description'])
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(new_rule)
                print(f"New rule '{rule_name}' added successfully.")
        except Exception as e:
            print(f"Error adding new rule to {csv_filename}: {e}")

    def censor_sentence(self, pos_tagged_sentence, profane_indices):
        return ' '.join(['*****'] * len(pos_tagged_sentence))

def predict_and_censor(sentence, pattern_generator, best_model, threshold=0.5):
    probas = best_model.predict_proba([sentence])[0]
    is_profane = probas[1] >= threshold
    print(f"SVM Prediction: {'Profane' if is_profane else 'Not Profane'} with probability: {probas[1]}")
    if is_profane:
        pos_tagged_sentence = pattern_generator.tag_sentence(sentence)
        censored_sentence = pattern_generator.censor_sentence(pos_tagged_sentence, [])
        return censored_sentence
    else:
        return sentence

def main():
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    predefined_rules_path = f"{base_path}/PATTERN_GENERATION/predefined_rules.csv"
    model_filename = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
    pattern_generator = PatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    
    sentence = "patalon ka talaga maski nuarin"
    pattern_generator.save_patterns_from_sentence(predefined_rules_path, sentence, "Profane sentence example")
    
    df = pd.read_csv('UsedDataset/dataset_bikol_sentence_profane.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['profane'], test_size=0.8, random_state=48)
    
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), SVC(probability=True))
    param_grid = {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'bikol_trained_profane_model.pkl')
    print("Model saved as 'bikol_trained_profane_model.pkl'")

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    print("Training Data Classification Report:")
    print(classification_report(y_train, y_train_pred))
    
    print("Testing Data Classification Report:")
    print(classification_report(y_test, y_test_pred))

    cm = confusion_matrix(y_test, y_test_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()
    
    pos_tagged_sentence = pattern_generator.tag_sentence(sentence)
    profane_patterns, profane_ngram_indices = pattern_generator.detect_profane_patterns(pos_tagged_sentence)
    
    if profane_patterns:
        censored_sentence = pattern_generator.censor_sentence(pos_tagged_sentence, profane_ngram_indices)
        print(f"Original Sentence: {sentence}")
        print(f"Censored Sentence: {censored_sentence}")
    else:
        print(f"No profanity detected in the sentence: {sentence}")

if __name__ == "__main__":
    main()
