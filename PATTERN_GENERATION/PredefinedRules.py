import csv
from collections import Counter
from nltk.util import ngrams
import os

class PatternGenerator:
    def __init__(self, csv_filename):
        # Load predefined rules from CSV
        self.rules = self.load_predefined_rules(csv_filename)
        
    def load_predefined_rules(self, csv_filename):
        """Load predefined profane rules from a CSV file."""
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
                        'POS Pattern': row['POS Pattern'].split(),  # Split POS pattern into list
                        'Description': row['Description']
                    })
        except Exception as e:
            print(f"Error loading rules from {csv_filename}: {e}")
        return rules
    
    def generate_ngrams(self, pos_tagged_text, n):
        """Generate N-Grams (unigrams, bigrams, trigrams) from 'word|POS' tagged text."""
        # Expecting a list of 'word|POS' strings
        if all(isinstance(item, str) and '|' in item for item in pos_tagged_text):
            tokens = [item.split('|')[1] for item in pos_tagged_text]  # Extract POS tags from 'word|POS'
        else:
            print(f"Received incorrect format: {pos_tagged_text}")  # Print the incorrect format
            raise ValueError("Expected input in the form of 'word|POS', but got incorrect format.")
        return list(ngrams(tokens, n))
    
    def apply_rules_to_ngrams(self, ngram_list):
        """Apply predefined rules to the list of N-Grams."""
        flagged_patterns = []

        for ngram in ngram_list:
            for rule in self.rules:
                pattern = tuple(rule['POS Pattern'])  # Corrected key
                if ngram == pattern:
                    flagged_patterns.append(f"Rule Matched: {rule['Rule Name']} - {rule['Description']}")
        
        return flagged_patterns if flagged_patterns else "No profane patterns detected"

    def detect_profane_patterns(self, pos_tagged_text):
        """Detect profane patterns in 'word|POS' tagged text using predefined rules and N-Gram models."""
        results = []
        
        # Check for unigrams, bigrams, and trigrams
        for n in range(1, 4):
            ngrams_list = self.generate_ngrams(pos_tagged_text, n)
            results += self.apply_rules_to_ngrams(ngrams_list)
        
        return results if results else "No profane patterns detected"

    def add_new_rule(self, csv_filename, rule_name, pos_pattern, description):
        """Add a new predefined rule to the CSV file."""
        new_rule = {
            'Rule Name': rule_name,
            'POS Pattern': pos_pattern,  # Store as string; split during reading
            'Description': description
        }
        try:
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['Rule Name', 'POS Pattern', 'Description'])
                writer.writerow(new_rule)
            print(f"New rule '{rule_name}' added successfully.")
        except Exception as e:
            print(f"Error adding new rule to {csv_filename}: {e}")
        
# Example usage
if __name__ == "__main__":
    # Set the correct path to the predefined rules CSV file
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    predefined_rules_path = f"{base_path}/PATTERN_GENERATION/predefined_rules.csv"
    
    # Initialize pattern generator with predefined rules CSV file
    pattern_generator = PatternGenerator(predefined_rules_path)
    
    # Example POS-tagged text (e.g., ['Tanga|JJD', 'mo|PRS', 'bobo|NNC'])
    pos_tagged_sentence = ['Tanga|JJD', 'mo|PRS', 'bobo|NNC', 'pakyu|FW']
    
    # Detect profane patterns in the POS-tagged sentence
    detected_patterns = pattern_generator.detect_profane_patterns(pos_tagged_sentence)
    
    print("Detected Profane Patterns:", detected_patterns)
    
    # Example of adding a new rule to the CSV file
    pattern_generator.add_new_rule(predefined_rules_path, 'new_adj_noun_rule', 'JJD PRS NNC', 'Profane Adjective-Pronoun-Noun')
