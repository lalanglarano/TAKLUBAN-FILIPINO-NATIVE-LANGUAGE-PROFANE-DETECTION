import pandas as pd
from nltk import ngrams
from collections import Counter
import csv
from PredefinedRules import PatternGenerator
from nltk.tag import StanfordPOSTagger

# Set the paths to the Stanford POS Tagger model and jar files
path_to_model = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar'

# Initialize the tagger with the model and jar paths
stanford_pos_tagger = StanfordPOSTagger(path_to_model, path_to_jar)

# Tag the sentence (example)
pos_tagged_sentence = stanford_pos_tagger.tag(['putangina', 'inaantok', 'ako'])

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

# Rule-based profanity detection function using predefined rules from PatternGenerator
def rule_based_profanity_detection(pos_tagged_ngram, pattern_generator):
    print(f"POS-tagged N-gram: {pos_tagged_ngram}")  # Debugging line

    # Ensure that the input is a list of 'word|POS' strings
    if not all(isinstance(item, str) and '|' in item for item in pos_tagged_ngram):
        raise ValueError(f"Expected list of 'word|POS' strings, but got {pos_tagged_ngram}")

    # Generate the POS pattern (e.g., "JJD NNC") from the 'word|POS' format
    pos_pattern = ' '.join([item.split('|')[1] for item in pos_tagged_ngram])
    
    # Detect profane patterns
    profane_patterns = pattern_generator.detect_profane_patterns(pos_tagged_ngram)
    if profane_patterns and profane_patterns != "No profane patterns detected":
        print(f"Matched profane pattern: {profane_patterns}")
        return True
    return False  

# Function to tag N-grams with POS tags
def pos_tag_ngrams(ngrams_list, pos_tagger):
    pos_tagged_ngrams = []
    for ngram in ngrams_list:
        # POS tag each word in the n-gram
        pos_tagged = pos_tagger.tag(list(ngram))  # Pass n-gram words as a list for tagging
        pos_tagged_ngrams.append(['{}|{}'.format(word, tag) for word, tag in pos_tagged])  # Create 'word|POS' format
    return pos_tagged_ngrams

# Updated function to handle potentially profane N-grams
def tag_and_apply_rules(potential_profanity, pattern_generator, pos_tagger):
    # POS tag each N-gram
    pos_tagged_ngrams = pos_tag_ngrams(potential_profanity, pos_tagger)
    
    for pos_tagged_ngram in pos_tagged_ngrams:
        # Print for debugging purposes
        print(f"POS-tagged N-gram: {pos_tagged_ngram}")
        
        if rule_based_profanity_detection(pos_tagged_ngram, pattern_generator):
            return True  # Profanity detected
    return False

# Example usage:
csv_file = 'PATTERN_GENERATION/predefined_rules.csv'
column_index = 0

# Load dataset from CSV
dataset = load_dataset(csv_file, column_index)

# Train the N-gram model
n = 2  # Using bigrams in this case
ngram_model = train_ngram_model(dataset, n)

# Define a threshold for profanity detection
frequency_threshold = 1  # Adjust as needed

# Test with a new sentence
new_sentence = "putangina inaantok ako"
print(f"Input Sentence: {new_sentence}")  # Debugging log
potential_profanity = detect_profanity(new_sentence, ngram_model, n, frequency_threshold)

# Initialize PatternGenerator with predefined rules CSV file
predefined_rules_path = 'PATTERN_GENERATION/predefined_rules.csv'
pattern_generator = PatternGenerator(predefined_rules_path)

# Display potentially profane N-grams
if potential_profanity:
    print("Potentially profane N-grams found:")
    
    # Apply custom rules for further filtering profane N-grams using predefined rules
    filtered_profanity = tag_and_apply_rules(potential_profanity, pattern_generator, stanford_pos_tagger)
    
    if filtered_profanity:
        print(f"Profanity detected: {filtered_profanity}")
    else:
        print("No profane N-grams detected after rule-based filtering.")
else:
    print("No potentially profane N-grams found.")
