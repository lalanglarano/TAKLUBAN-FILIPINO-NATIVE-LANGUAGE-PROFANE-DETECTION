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

# Function to load dataset from a CSV file with a specific column
def load_dataset(csv_file, column_name):
    """
    Loads a dataset from a CSV file using a column name.
    """
    try:
        data = pd.read_csv(csv_file)  # Ensure CSV is loaded
        sentences = data[column_name].dropna().tolist()  # Access column by name, dropping empty rows
        return sentences
    except KeyError:
        raise KeyError(f"Column {column_name} not found in the CSV file.")

# Function to generate N-grams from tokens
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Function to train an N-gram model on a dataset
def train_ngram_model(dataset, n=3):
    """
    Trains an N-gram model by counting the frequencies of N-grams in the dataset.
    """
    ngram_freq = Counter()
    for sentence in dataset:
        tokens = sentence.split()  # Split sentences into words
        ngrams_list = generate_ngrams(tokens, n)
        ngram_freq.update(ngrams_list)
    return ngram_freq

# Function to detect potentially profane N-grams based on frequency threshold
def detect_profanity(sentence, ngram_model, n=3, threshold=1):
    """
    Detects potentially profane N-grams by checking their frequency.
    """
    tokens = sentence.split()  # Assume input sentence is pre-tokenized
    ngrams_list = generate_ngrams(tokens, n)
    
    potential_profanity = []
    for ngram in ngrams_list:
        if ngram_model.get(ngram, 0) <= threshold:  # Check if frequency is below threshold
            potential_profanity.append(ngram)
    
    print(f"Detected N-grams: {potential_profanity}")  # Debugging log
    return potential_profanity

# Rule-based profanity detection function using predefined rules from PatternGenerator
def rule_based_profanity_detection(pos_tagged_ngram, pattern_generator):
    print(f"POS-tagged N-gram: {pos_tagged_ngram}")  # Debugging line

    # Ensure that the input is a list of 'word|POS' strings
    if not all(isinstance(item, str) and '|' in item for item in pos_tagged_ngram):
        raise ValueError(f"Expected list of 'word|POS' strings, but got {pos_tagged_ngram}")

    # Generate the POS pattern (e.g., "RBW VBTR") from the 'word|POS' format
    pos_pattern = ' '.join([item.split('|')[1] for item in pos_tagged_ngram])
    
    # Detect profane patterns
    profane_patterns = pattern_generator.detect_profane_patterns(pos_tagged_ngram)  # Send entire 'word|POS' list

    if profane_patterns and profane_patterns != "No profane patterns detected":
        print(f"Matched profane pattern: {profane_patterns}")
        # Return the n-gram with POS-tagged pairs enclosed in <>
        profane_ngram = ' '.join([f"<{item}>" for item in pos_tagged_ngram])
        return profane_ngram
    return False

# Function to tag N-grams with POS tags
def pos_tag_ngrams(ngrams_list, pos_tagger):
    """
    POS tags each word in the detected N-grams.
    """
    pos_tagged_ngrams = []
    for ngram in ngrams_list:
        pos_tagged = pos_tagger.tag(list(ngram))  # Pass n-gram words as a list for tagging
        pos_tagged_ngrams.append(['{}|{}'.format(word, tag) for word, tag in pos_tagged])  # Create 'word|POS' format
    return pos_tagged_ngrams

# Updated function to handle potentially profane N-grams
def tag_and_apply_rules(potential_profanity, pattern_generator, pos_tagger):
    """
    Tags the N-grams and applies the predefined rules to detect profane patterns.
    """
    pos_tagged_ngrams = pos_tag_ngrams(potential_profanity, pos_tagger)
    
    modified_sentence = ""
    for pos_tagged_ngram in pos_tagged_ngrams:
        print(f"POS-tagged N-gram: {pos_tagged_ngram}")  # Debugging
        
        profane_ngram = rule_based_profanity_detection(pos_tagged_ngram, pattern_generator)
        if profane_ngram:
            modified_sentence += f"{profane_ngram} "
        else:
            # Append original POS-tagged N-gram if no profanity detected
            original_ngram = ' '.join([f"<{word}|{tag}>" for word, tag in [item.split('|') for item in pos_tagged_ngram]])
            modified_sentence += f"{original_ngram} "

    return modified_sentence.strip()  # Return the modified sentence

# Example usage:
csv_file = 'Results/pos_tagged/FPOSTagged_tagalog.csv'
column_name = 'pos_tagged'  # Adjust this if the column name is different

# Load dataset from CSV
dataset = load_dataset(csv_file, column_name)

# Train the N-gram model
n = 2  # Using bigrams in this case
ngram_model = train_ngram_model(dataset, n)

# Define a threshold for profanity detection
frequency_threshold = 2  # Adjust as needed

# Test with a new sentence
new_sentence = "mapagalon na yudeputa"
print(f"Input Sentence: {new_sentence}")  # Debugging log
potential_profanity = detect_profanity(new_sentence, ngram_model, n, frequency_threshold)

# Initialize PatternGenerator with predefined rules CSV file
predefined_rules_path = 'PATTERN_GENERATION/predefined_rules.csv'
pattern_generator = PatternGenerator(predefined_rules_path)

# Display potentially profane N-grams
if potential_profanity:
    print("Potentially profane N-grams found:")
    
    # Apply custom rules for further filtering profane N-grams using predefined rules
    modified_sentence = tag_and_apply_rules(potential_profanity, pattern_generator, stanford_pos_tagger)
    
    if modified_sentence:
        print(f"Final sentence with POS-tagged N-grams enclosed: {modified_sentence}")
    else:
        print("No profane N-grams detected after rule-based filtering.")
else:
    print("No potentially profane N-grams found.")
