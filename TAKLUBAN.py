import csv
import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from LanguageIdentification.FNLI import LanguageIdentification
from PATTERN_GENERATION.tagalog import PatternGenerator as TagalogPatternGenerator
from PATTERN_GENERATION.bikol import PatternGenerator as BikolPatternGenerator
from PATTERN_GENERATION.cebuano import PatternGenerator as CebuanoPatternGenerator

# Configuration for paths
output_file = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/POSdata.csv"
profanity_dictionary_file = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/profanity_dictionary.csv"
model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"

# Create CSV headers if they do not exist
def initialize_csv_files():
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Language', 'Sentence', 'POS', 'Censored Sentence'])

    if not os.path.exists(profanity_dictionary_file):
        with open(profanity_dictionary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Language', 'Profane Sentence'])

initialize_csv_files()

# Define noise words for multiple languages
noise_words = {
    "na", "nang", "ng", "mga", "ang", "kung", "yan", "ito", "si", "ko", "po", "ka", "ikaw", "siya", "oo",
    "sa", "may", "ni", "dahil", "kasi", "pero", "at", "para", "niya", "saan", "ganito", "doon", "noon", 
    "ta", "ngani", "ini", "kang", "iyo", "hali", "baga", "ho", "mo", "ba", "si", "kan", "kun", "ngani",
    "yan", "sadi", "pala", "yaon", "ini", "yan", "na", "digdi", "dakol", "bangan", "dayon", "ang", "ini",
    "gani", "kana", "mao", "pud", "bitaw", "ta", "si", "ug", "naa", "dili", "kini", "adto", "man", "kay",
    "unta", "nga", "sa", "kani", "mo", "lang", "sila", "unsa", "ako", "niyo", "su"
}

# Save sentence data and patterns to CSV
def save_to_csv(language, sentence, pos_tagged, censored_sentence):
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([language, sentence, pos_tagged, censored_sentence])

def save_profane_pattern(language, sentence, pos_pattern):
    with open(profanity_dictionary_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([language, sentence, pos_pattern])

# Train model if it doesn't exist
def train_model_if_not_exists(model_path, dictionary_dir):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Training a new model...")
        trainer = LanguageIdentification(config)  # Assuming `config` is defined elsewhere
        model, X_test, y_test = trainer.train_model()
        joblib.dump(model, model_path)
        print(f"Model trained and saved at {model_path}.")
        return model, X_test, y_test
    else:
        print(f"Loading pre-saved model from {model_path}.")
        model = joblib.load(model_path)
        return model, [], []

# Get appropriate pattern generator based on language
def get_pattern_generator(language):
    predefined_rules_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/PATTERN_GENERATION/predefined_rules.csv"
    model_filename = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar' 
    
    pattern_generators = {
        "tagalog": TagalogPatternGenerator(predefined_rules_path, model_filename, path_to_jar),
        "bikol": BikolPatternGenerator(predefined_rules_path, model_filename, path_to_jar),
        "cebuano": CebuanoPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    }
    
    return pattern_generators.get(language.lower())

# Process and censor sentences
def predict_and_censor(sentence, pattern_generator, model, language):
    pos_tagged_sentence = pattern_generator.tag_sentence(sentence)
    tokens = sentence.split()
    filtered_tokens = [token for token in tokens if token.lower() not in noise_words]
    
    if not filtered_tokens:
        return sentence, False

    censored_tokens = []
    is_profane = False
    for token in tokens:
        if token.lower() in noise_words:
            censored_tokens.append(token)
        else:
            if model.predict([token])[0] == 1:
                censored_tokens.append(re.sub(r'\w', '*', token))
                is_profane = True
            else:
                censored_tokens.append(token)

    censored_sentence = ' '.join(censored_tokens)
    if is_profane:
        pos_pattern = ' '.join([item.split('\n')[-1] for item in pos_tagged_sentence if '\n' in item])
        save_profane_pattern(language, sentence, pos_pattern)
    return censored_sentence, is_profane

# Display output for the user
def display_output(language, sentence, pos_tagged, censored_sentence, is_profane):
    print(f"\nDetected Language: {language}")
    print(f"Original Sentence: {sentence}")
    print(f"POS Tagged Sentence: {pos_tagged}")
    if is_profane:
        print(f"Censored Sentence: {censored_sentence} (Profane)")
    else:
        print(f"Cleaned Sentence: {censored_sentence} (Not Profane)")
    save_to_csv(language, sentence, pos_tagged, censored_sentence)
    print(f"Sentence '{sentence}' saved with the detected language and POS tagging result.\n")

# Main function for running the process
def main():
    model, _, _ = train_model_if_not_exists(model_path, "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary")
    
    while True:
        sentence = input("Enter a sentence for language detection and profanity check: ").strip()
        if not sentence:
            break
        
        # Predict language
        language = model.predict([sentence])[0].lower()
        print(f"Predicted language: {language.capitalize()}")
        
        # Get appropriate pattern generator
        pattern_generator = get_pattern_generator(language)
        
        # Censor profane words
        censored_sentence, is_profane = predict_and_censor(sentence, pattern_generator, model, language)
        
        # Display results
        display_output(language, sentence, "", censored_sentence, is_profane)

if __name__ == "__main__":
    main()
