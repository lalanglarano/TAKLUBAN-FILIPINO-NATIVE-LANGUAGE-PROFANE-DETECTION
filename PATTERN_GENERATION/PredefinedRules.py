import csv
import os
from nltk.util import ngrams

class PatternGenerator:
    def __init__(self, csv_filename):
        self.rules = self.load_predefined_rules(csv_filename)
        
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
            tokens = [item.split('|')[1] for item in pos_tagged_text]
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
        return results if results else ["No profane patterns detected"]

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

if __name__ == "__main__":
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    predefined_rules_path = f"{base_path}/PATTERN_GENERATION/predefined_rules.csv"
    
    pattern_generator = PatternGenerator(predefined_rules_path)
    
    pos_tagged_sentence = ['Tanga|JJD', 'mo|PRS', 'bobo|NNC', 'pakyu|FW']
    
    detected_patterns = pattern_generator.detect_profane_patterns(pos_tagged_sentence)
    print("Detected Profane Patterns:", detected_patterns)
    
    pattern_generator.add_new_rule(predefined_rules_path, 'new_adj_noun_rule', 'JJD PRS NNC', 'Profane Adjective-Pronoun-Noun')
