import os
import pandas as pd
from nltk import ngrams
from collections import Counter
from PredefinedRules import PatternGenerator
from nltk.tag import StanfordPOSTagger

# Set the paths to the Stanford POS Tagger model and jar files
path_to_model = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar'

# Initialize the tagger with the model and jar paths
stanford_pos_tagger = StanfordPOSTagger(path_to_model, path_to_jar)

# Function to load dataset from a CSV file with a specific column
def load_dataset(csv_file, column_name):
    try:
        data = pd.read_csv(csv_file)
        sentences = data[column_name].dropna().tolist()
        return sentences
    except KeyError:
        raise KeyError(f"Column {column_name} not found in the CSV file.")

# Function to generate N-grams from tokens
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Function to train an N-gram model on a dataset
def train_ngram_model(dataset, n=3):
    ngram_freq = Counter()
    for sentence in dataset:
        tokens = sentence.split()
        ngrams_list = generate_ngrams(tokens, n)
        ngram_freq.update(ngrams_list)
    return ngram_freq

# Function to detect potentially profane N-grams based on frequency threshold
def detect_profanity(sentence, ngram_model, n=3, threshold=1):
    tokens = sentence.split()
    ngrams_list = generate_ngrams(tokens, n)
    
    potential_profanity = []
    for ngram in ngrams_list:
        if ngram_model.get(ngram, 0) <= threshold:
            potential_profanity.append(ngram)
    
    print(f"Detected N-grams: {potential_profanity}")
    return potential_profanity

# Rule-based profanity detection function using predefined rules from PatternGenerator
def rule_based_profanity_detection(pos_tagged_ngram, pattern_generator, predefined_rules_file):
    print(f"POS-tagged N-gram: {pos_tagged_ngram}")

    if not all(isinstance(item, str) and '|' in item for item in pos_tagged_ngram):
        raise ValueError(f"Expected list of 'word|POS' strings, but got {pos_tagged_ngram}")

    pos_pattern = ' '.join([item.split('|')[1] for item in pos_tagged_ngram])
    profane_patterns = pattern_generator.detect_profane_patterns(pos_tagged_ngram)

    if profane_patterns and profane_patterns != "No profane patterns detected":
        print(f"Matched profane pattern: {profane_patterns}")
        profane_ngram = ' '.join([f"<{item}>" for item in pos_tagged_ngram])

        # Save the newly detected pattern to the predefined rules CSV using add_new_rule from PatternGenerator
        rule_name = "generated_rule"
        description = "Automatically detected profane pattern"
        pattern_generator.add_new_rule(predefined_rules_file, rule_name, pos_pattern, description)

        return profane_ngram
    return False

# Function to tag N-grams with POS tags
def pos_tag_ngrams(ngrams_list, pos_tagger):
    pos_tagged_ngrams = []
    for ngram in ngrams_list:
        pos_tagged = pos_tagger.tag(list(ngram))
        pos_tagged_ngrams.append(['{}|{}'.format(word, tag) for word, tag in pos_tagged])
    return pos_tagged_ngrams

# Updated function to handle potentially profane N-grams
def tag_and_apply_rules(potential_profanity, pattern_generator, pos_tagger, predefined_rules_file):
    pos_tagged_ngrams = pos_tag_ngrams(potential_profanity, pos_tagger)
    
    modified_sentence = ""
    for pos_tagged_ngram in pos_tagged_ngrams:
        print(f"POS-tagged N-gram: {pos_tagged_ngram}")

        profane_ngram = rule_based_profanity_detection(pos_tagged_ngram, pattern_generator, predefined_rules_file)
        if profane_ngram:
            modified_sentence += f"{profane_ngram} "
        else:
            original_ngram = ' '.join([f"<{item}>" for item in pos_tagged_ngram])
            modified_sentence += f"{original_ngram} "

    return modified_sentence.strip()

# Example usage:
csv_file = 'Results/pos_tagged/FPOSTagged_tagalog.csv'
column_name = 'pos_tagged'

# Load dataset from CSV
dataset = load_dataset(csv_file, column_name)

# Train the N-gram model
n = 2
ngram_model = train_ngram_model(dataset, n)

# Define a threshold for profanity detection
frequency_threshold = 2

# Test with a new sentence
new_sentence = "san ka na naman puputang punyeta ka"
print(f"Input Sentence: {new_sentence}")
potential_profanity = detect_profanity(new_sentence, ngram_model, n, frequency_threshold)

# Initialize PatternGenerator with predefined rules CSV file
predefined_rules_path = 'PATTERN_GENERATION/predefined_rules.csv'

# Initialize the PatternGenerator with the predefined rules
pattern_generator = PatternGenerator(predefined_rules_path)

# Display potentially profane N-grams
if potential_profanity:
    print("Potentially profane N-grams found:")
    modified_sentence = tag_and_apply_rules(potential_profanity, pattern_generator, stanford_pos_tagger, predefined_rules_path)
    
    if modified_sentence:
        print(f"Final sentence with POS-tagged N-grams enclosed: {modified_sentence}")
    else:
        print("No profane N-grams detected after rule-based filtering.")
else:
    print("No potentially profane N-grams found.")
