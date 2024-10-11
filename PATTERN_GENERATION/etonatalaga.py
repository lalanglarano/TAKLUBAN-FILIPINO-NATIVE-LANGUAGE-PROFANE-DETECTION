import os
import pandas as pd
import csv
from nltk.tag.stanford import StanfordPOSTagger
from nltk.util import ngrams
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

class PatternGenerator:
    def __init__(self, csv_filename, model_filename, path_to_jar):
        self.rules = self.load_predefined_rules(csv_filename)
        self.tagger = StanfordPOSTagger(model_filename=model_filename, path_to_jar=path_to_jar)
        
    def load_predefined_rules(self, csv_filename):
        rules = []
        if not os.path.exists(csv_filename):
            print(f"Error: {csv_filename} does not exist.")
            return rules
        
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
        if all(isinstance(item, str) and '|' in item for item in pos_tagged_text):
            tokens = [item.split('|') for item in pos_tagged_text]
        else:
            raise ValueError("Expected input in the form of 'word|POS', but got incorrect format.")
        return list(ngrams(tokens, n))
    
    def apply_rules_to_ngrams(self, ngram_list):
        flagged_patterns = []
        for ngram in ngram_list:
            for rule in self.rules:
                pattern = tuple(rule['POS Pattern'])
                if ngram == pattern:
                    flagged_patterns.append(f"Rule Matched: {rule['Rule Name']} - {rule['Description']}")
        return flagged_patterns
    
    def detect_profane_patterns(self, pos_tagged_text):
        results = []
        for n in range(1, 4):
            ngrams_list = self.generate_ngrams(pos_tagged_text, n)
            results += self.apply_rules_to_ngrams(ngrams_list)
        
        return results if results else "No profane patterns detected"
    
    def add_new_rule(self, csv_filename, rule_name, pos_pattern, description):
        current_rules = self.load_predefined_rules(csv_filename)
        for rule in current_rules:
            if rule['Rule Name'] == rule_name and ' '.join(rule['POS Pattern']) == pos_pattern:
                print(f"Rule '{rule_name}' with POS pattern '{pos_pattern}' already exists.")
                return
        
        new_rule = {
            'Rule Name': rule_name,
            'POS Pattern': pos_pattern,
            'Description': description
        }
        
        try:
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['Rule Name', 'POS Pattern', 'Description'])
                writer.writerow(new_rule)
            print(f"New rule '{rule_name}' added successfully.")
        except Exception as e:
            print(f"Error adding new rule to {csv_filename}: {e}")
    
    def tag_sentence(self, sentence):
        tokens = sentence.split()
        tagged_sentence = self.tagger.tag(tokens)
        return [f"{word}|{tag}" for word, tag in tagged_sentence]
    
    def save_patterns_from_sentence(self, csv_filename, sentence, description):
        pos_tagged_sentence = self.tag_sentence(sentence)
        pos_pattern = ' '.join([tag.split('|')[2] for tag in pos_tagged_sentence])
        rule_name = f"rule_from_sentence_{len(self.rules) + 1}"
        self.add_new_rule(csv_filename, rule_name, pos_pattern, description)

def main():
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    predefined_rules_path = f"{base_path}/PATTERN_GENERATION/predefined_rules.csv"
    model_filename = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
    
    pattern_generator = PatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    
    # Load your dataset
    df = pd.read_csv('UsedDataset/dataset_tagalog_sentence_profane.csv')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['profane'], test_size=0.5, random_state=48)
    
    # Create a pipeline that combines the TfidfVectorizer with N-Grams and SVM
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

    def predict_and_censor(sentence):
        pos_tagged_sentence = pattern_generator.tag_sentence(sentence)
        print("POS-tagged Sentence:", pos_tagged_sentence)
        
        detected_patterns = pattern_generator.detect_profane_patterns(pos_tagged_sentence)
        print("Detected Patterns:", detected_patterns)
        
        # Extract words tagged as profane
        profane_words = set()
        for pattern in detected_patterns:
            if 'Rule Matched' in pattern:
                matched_rule = pattern.split(': ')[1].split(' - ')[0]
                print("Matched Rule:", matched_rule)
                for word_tag in pos_tagged_sentence:
                    word, tag = word_tag.rsplit('|', 1)
                    for rule_word in matched_rule.split():
                        if rule_word in word:
                            profane_words.add(word)
                            print("Profane Word Detected:", word)
        
        # Create the result sentence with censored profane words
        result_sentence = ' '.join(
            f"{word}<{tag}>" if word not in profane_words else "*****<profane>"
            for word, tag in [item.rsplit('|', 1) for item in pos_tagged_sentence]
        )
        
        return result_sentence

    # Example usage
    sentence = "tangina mo ka"
    prediction = predict_and_censor(sentence)
    print(f"Output: {prediction}")

if __name__ == "__main__":
    main()
