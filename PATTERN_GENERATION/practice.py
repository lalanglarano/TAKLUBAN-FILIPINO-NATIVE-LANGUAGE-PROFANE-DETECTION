import os
import pandas as pd
from nltk import ngrams
from collections import Counter
import csv

# Function to load dataset from a CSV file without column names
def load_dataset(csv_file, column_index):
    data = pd.read_csv(csv_file, header=None)
    sentences = data[column_index].dropna().tolist()
    return sentences

# Function to generate N-grams from tokens
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Function to train an N-gram model on a dataset
def train_ngram_model(dataset, n=3):
    ngram_freq = Counter()
    for sentence in dataset:
        tokens = sentence.split()  # Assume the sentences are pre-tokenized
        ngrams_list = generate_ngrams(tokens, n)
        ngram_freq.update(ngrams_list)
    return ngram_freq

# Function to detect potentially profane N-grams based on frequency threshold
def detect_profanity(sentence, ngram_model, n=3, threshold=1):
    tokens = sentence.split()  # Assume input sentence is pre-tokenized
    ngrams_list = generate_ngrams(tokens, n)
    
    potential_profanity = []
    for ngram in ngrams_list:
        if ngram_model.get(ngram, 0) <= threshold:  # Default frequency is 0 if not found
            potential_profanity.append(ngram)
    
    print(f"Detected N-grams: {potential_profanity}")  # Debugging log
    return potential_profanity

# Function to load POS patterns from a CSV file
def load_pos_patterns(csv_file):
    patterns = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            patterns.append(row)  # Each row is a pattern (list of POS tags)
    return patterns

# Rule-based profanity detection function using POS patterns from a CSV file
def rule_based_profanity_detection(pos_tagged_ngram, patterns):
    # Extract POS tags from tagged N-grams
    pos_tags = [tag for tag in pos_tagged_ngram]
    
    # Check if the n-gram matches any of the patterns from the CSV
    for pattern in patterns:
        if pos_tags[:len(pattern)] == pattern:
            return True
    
    # Additional check for suspicious word contexts (e.g., rare verbs followed by aggressive words)
    if len(pos_tags) > 1 and ("VBTS" in pos_tags or "JJCM" in pos_tags):  # Example context check
        # Look for combinations of verbs and aggressive-sounding adjectives or interjections
        if "PRS" in pos_tags or "NNC" in pos_tags:
            return True
    
    print(f"No match for N-gram: {pos_tags}")  # Debugging log
    return False

# Function to simulate POS tagging and applying custom rules
def tag_and_apply_rules(potential_profanity, patterns):
    detected_profanity = []
    
    # Simulated POS tags for known words (you can adjust this to your real POS tagging later)
    simulated_pos_tags = {
        'putangina': ['EXPR'],  # Verb
        'inaantok': ['VB'],         # Conjunction
        'ako': ['PRS'],        # Pronoun
        'agad': ['ADV'],    # Noun in a profane context (you can choose another tag like 'PROFANE')
    }
    
    for ngram in potential_profanity:
        pos_tags = []
        for word in ngram:
            # Simulate POS tagging for each word
            pos_tags.extend(simulated_pos_tags.get(word, ['UNK']))  # 'UNK' for unknown tags
        
        # Apply a rule-based approach to check if this n-gram matches profane patterns
        if rule_based_profanity_detection(pos_tags, patterns):
            detected_profanity.append(ngram)
        else:
            # Check if any single word in the N-gram is profane (e.g., "punyeta" by itself)
            for word in ngram:
                if word in simulated_pos_tags and simulated_pos_tags[word] == ['EXPR']: 
                    print(f"Standalone profane word detected: {word}")
                    detected_profanity.append(ngram)
                    break
    
    print(f"Filtered Profanity: {detected_profanity}")  # Debugging log
    return detected_profanity

# Example usage:
csv_file = 'Results/pos_tagged/FPOSTagged_tagalog.csv'
column_index = 0

# Load dataset from CSV
dataset = load_dataset(csv_file, column_index)

# Train the N-gram model
n = 2  # Using bigrams in this case
ngram_model = train_ngram_model(dataset, n)

# Define a threshold for profanity detection
frequency_threshold = 1  # Adjust as needed

# Test with a new sentence
new_sentence = "putangina inaantok ako agad"
print(f"Input Sentence: {new_sentence}")  # Debugging log
potential_profanity = detect_profanity(new_sentence, ngram_model, n, frequency_threshold)

# Load POS patterns from CSV
csv_file_patterns = 'Results/ExtractedPOSonly/tagalog/unique_pos_patterns.csv'  # Adjust this path to your actual CSV file
patterns = load_pos_patterns(csv_file_patterns)

# Display potentially profane N-grams
if potential_profanity:
    print("Potentially profane N-grams found:")
    
    # Apply custom rules for further filtering profane N-grams
    filtered_profanity = tag_and_apply_rules(potential_profanity, patterns)
    
    if filtered_profanity:
        print(f"Profanity detected: {filtered_profanity}")
    else:
        print("No profane N-grams detected after rule-based filtering.")
else:
    print("No potentially profane N-grams found.")
