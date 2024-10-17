import os
import pandas as pd
import csv
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from nltk.util import ngrams
from POSTagging.POSTAGGER.pospkl.POSTagger import POSTagger 

class PatternGenerator:
    def __init__(self, csv_filename, language='cebuano'):
        self.rules = self.load_predefined_rules(csv_filename)
        self.tagger = POSTagger(language)

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
        # Reuse the POS extraction code
        pos_tags = [item.split('|')[-1] for item in pos_tagged_text if '|' in item]  # Extract POS tags only
        print(f"Extracted POS Tags: {pos_tags}")  # Debugging: Print extracted POS tags

        # Generate n-grams using the extracted POS tags
        ngrams_list = list(ngrams(pos_tags, n))
        print(f"Generated {n}-grams: {ngrams_list}")  # Debugging: Print generated n-grams
        return ngrams_list

    def apply_rules_to_ngrams(self, ngram_list):
        flagged_patterns = []
        matching_ngram_indices = []  # Store indices of the profane n-grams

        for idx, ngram in enumerate(ngram_list):
            print(f"Checking n-gram: {ngram}")  # Debugging: Output the generated n-gram
            for rule in self.rules:
                pattern = rule['POS Pattern']  # The POS Pattern is already a list of POS tags

                # Ensure the n-gram and pattern are of the same length before comparison
                if len(ngram) == len(pattern) and list(ngram) == pattern:
                    flagged_patterns.append(f"Rule Matched: {rule['Rule Name']} - {rule['Description']}")
                    matching_ngram_indices.append(idx)  # Store the index of the matched n-gram
                    print(f"Match found: {rule['Rule Name']}")  # Debugging

        return flagged_patterns, matching_ngram_indices

    def detect_profane_patterns(self, pos_tagged_text):
        results = []
        profane_ngram_indices = []  # Add this to track the indices of matched n-grams

        # Loop over n-gram lengths (1-gram, 2-gram, 3-gram)
        for n in range(1, 11):  # For 1-gram, 2-gram, 3-gram
            # Generate n-grams from the POS-tagged sentence
            ngrams_list = self.generate_ngrams(pos_tagged_text, n)

            # Apply the predefined rules to the generated n-grams
            detected_patterns, ngram_indices = self.apply_rules_to_ngrams(ngrams_list)

            # If patterns are detected, add them to the results and track the n-gram indices
            if detected_patterns:
                results += detected_patterns
                profane_ngram_indices += ngram_indices

        # Return the detected patterns and the indices of the profane n-grams
        return results if results else ["No profane patterns detected"], profane_ngram_indices

    def save_patterns_from_sentence(self, csv_filename, sentence, description):
        """
        This method tags the sentence with POS tags, extracts the POS pattern,
        and saves the pattern as a rule in the CSV file.
        """
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
        """
        Use the POSTagger from POSTagger.py to tag the sentence.
        """
        pos_tagged_text = self.tagger.pos_tag_text(sentence)  # Use pos_tag_text instead of tag
        return pos_tagged_text.split()  # Return the tagged tokens as a list

    def add_new_rule(self, csv_filename, rule_name, pos_pattern, description):
        """
        Adds a new rule to the CSV file with the POS pattern and description.
        """
        current_rules = self.load_predefined_rules(csv_filename)
        for rule in current_rules:
            if rule['POS Pattern'] == pos_pattern.split():
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
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(new_rule)
                print(f"New rule '{rule_name}' added successfully.")
        except Exception as e:
            print(f"Error adding new rule to {csv_filename}: {e}")

    def censor_sentence(self, pos_tagged_sentence, profane_indices):
        """Censor the entire sentence if any profane pattern is detected."""
        censored_sentence = []
        for idx, word in enumerate(pos_tagged_sentence):
            censored_sentence.append('*****')  # Censor the entire sentence
        return ' '.join(censored_sentence)

# Main function remains unchanged
def main():
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    predefined_rules_path = f"{base_path}/PATTERN_GENERATION/predefined_rules.csv"

    # Initialize PatternGenerator with POSTagger
    pattern_generator = PatternGenerator(predefined_rules_path, language='cebuano')

    # Define the sentence to test
    sentence = "kolera ka bai"
        
    # Save pattern from the sentence
    pattern_generator.save_patterns_from_sentence(predefined_rules_path, sentence, "Profane sentence example")
    
    # Load your dataset
    df = pd.read_csv('UsedDataset/dataset_cebuano_sentence_profane.csv')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['profane'], test_size=0.8, random_state=48)
    
    # Create a pipeline that combines the TfidfVectorizer with N-Grams and SVM
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), SVC(probability=True))

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

    # Save the trained model to a .pkl file using joblib
    joblib.dump(best_model, 'cebuano_trained_profane_model.pkl')
    print("Model saved as 'cebuano_trained_profane_model.pkl'")

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Example usage
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
