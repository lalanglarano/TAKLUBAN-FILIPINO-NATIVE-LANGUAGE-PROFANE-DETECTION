"""
Program Title: Profanity Detection and Censorship

Programmers: Jeo Abarre, Annalyn Belen, Telish Gonzales, Randolph Larano

Where the program fits in the general system designs: This module detects and censors profane language in various Filipino languages.

Date written and last revised: October 5, 2024   |   November 18, 2024

Purpose: To detect and censor profane language in Tagalog, Bikol, and Cebuano using predefined rules 
and POS tagging. This module loads predefined rules, tags sentences with POS tags, detects profane 
patterns, censors detected patterns, and saves the results to a CSV file.

Data structures, algorithms, and control:

Data Structures:
output_file: Path to the CSV file for saving results.
profanity_dictionary_file: Path to the CSV file for saving profane patterns.
predictions, true_labels: Lists for storing predictions and true labels.
noise_words: A set of common noise words.

Algorithms:
Uses StanfordPOSTagger for POS tagging.
Uses predefined rules for detecting profane patterns.
Uses confusion matrix and classification report for performance evaluation.

Control:
Initializes with paths to output files and models.
Loads and preprocesses text data.
Tags sentences with POS tags.
Detects and censors profane patterns.
Saves results to CSV files.
Evaluates model performance using confusion matrix and classification report.
"""

import csv
import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import re
from nltk import ngrams
from collections import Counter
from sklearn.metrics import accuracy_score
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from LanguageIdentification.FNLI import LanguageIdentification, ModelTraining
from PATTERN_GENERATION.tagalog import PatternGenerator as TagalogPatternGenerator
from PATTERN_GENERATION.bikol import PatternGenerator as BikolPatternGenerator
from PATTERN_GENERATION.cebuano import PatternGenerator as CebuanoPatternGenerator

# Define the path to save the results
output_file = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/POSdata.csv"
profanity_dictionary_file = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/profanity_dictionary.csv"

# Check if the CSV file already exists. If not, create it and add a header
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Language', 'Sentence', 'POS', 'Censored Sentence'])  # Header for CSV

# Check if the profanity dictionary exists. If not, create it with a header
if not os.path.exists(profanity_dictionary_file):
    with open(profanity_dictionary_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Language', 'Profane Sentence'])  # Header for CSV

# Initialize lists to store predictions and true labels
predictions = []
true_labels = []

# Define noise words
noise_words = {
    "na", "nang", "ng", "mga", "ang", "kung", "yan", "ito", "si", "ko", "po", "ka", "ikaw", "siya", "oo",
    "sa", "may", "ni", "dahil", "kasi", "pero", "at", "para", "niya", "saan", "ganito", "doon", "noon", 
    "ta", "ngani", "ini", "kang", "iyo", "hali", "baga", "ho", "mo", "ba", "si", "kan", "kun", "ngani",
    "yan", "sadi", "pala", "yaon", "ini", "yan", "na", "digdi", "dakol", "bangan", "dayon", "ang", "ini",
    "gani", "kana", "mao", "pud", "bitaw", "ta", "si", "ug", "naa", "dili", "kini", "adto", "man", "kay",
    "eh", "naman", "kayo", "boi", "ika", "daw", "mag", "nag", "sako", "pa", "jan", "yon", "gabos", "yung",
    "unta", "nga", "sa", "kani", "mo", "lang", "sila", "unsa", "ako", "niyo", "su", "kita", "taka", "buda", "talaga"
}

def save_to_csv(language, sentence, pos_tagged, censored_sentence):
    """Save the language, sentence, POS tagged result, and censored sentence to a CSV file."""
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([language, sentence, pos_tagged, censored_sentence])

def save_profane_pattern(language, sentence, pos_pattern):
    """Save the language, profane sentence, and POS pattern to the profanity dictionary."""
    with open(profanity_dictionary_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([language, sentence, pos_pattern])

def train_model_if_not_exists(model_path, dictionary_dir):
    """Train and save the model if the pre-saved model is not found."""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Training a new model...")
        
        # Run the model training process
        trainer = ModelTraining(dictionary_dir)
        model, X_test, y_test = trainer.train_model()
        # Save the trained model
        joblib.dump(model, model_path)
        print(f"Model trained and saved at {model_path}.")
        return model, X_test, y_test
    else:
        # If model exists, load it
        print(f"Loading pre-saved model from {model_path}.")
        model = joblib.load(model_path)
        return model, [], []
          
def get_pattern_generator(language):
    print(f"Loading pattern generator for language: {language}")
    predefined_rules_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/PATTERN_GENERATION/predefined_rules.csv"
    model_filename = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar' 
    if language == 'tagalog':
        return TagalogPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    elif language == 'bikol':
        return BikolPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    elif language == 'cebuano': 
        return CebuanoPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    else:
        print(f"No pattern generator available for {language}")
        return None

def predict_and_censor(sentence, pattern_generator, model, language):
    """Perform profanity detection and censorship using the provided model, excluding noise words."""
    pos_tagged_sentence = pattern_generator.tag_sentence(sentence)
    tokens = sentence.split()

    # Filter tokens to exclude noise words before passing to the model
    filtered_tokens = [token for token in tokens if token.lower() not in noise_words]
    
    # If only noise words are left, return the sentence as-is without censorship
    if not filtered_tokens:
        print("Sentence contains only noise words; skipping censorship.")
        return sentence, False

    censored_tokens = []
    is_profane = False
    
    for token in tokens:
        if token.lower() in noise_words:
            censored_tokens.append(token)  # Keep noise words as-is
        else:
            # Predict only on non-noise words
            if model.predict([token])[0] == 1:  # Assuming 1 indicates a profane word
                censored_tokens.append(re.sub(r'\w', '*', token))
                is_profane = True
            else:
                censored_tokens.append(token)

    censored_sentence = ' '.join(censored_tokens)

    if is_profane:
        pos_pattern = ' '.join([item.split('\n')[-1] for item in pos_tagged_sentence if '\n' in item])
        save_profane_pattern(language, sentence, pos_pattern)
        print("Profane pattern saved to dictionary.")

    return censored_sentence, is_profane


def main():
    model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"
    dictionary_dir = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary"
    # Train the language identification model if it doesn't exist
    model, X_test, y_test = train_model_if_not_exists(model_path, dictionary_dir)
    # Initialize the LanguageIdentification class with the loaded model
    language_identifier = LanguageIdentification(model=model, X_test=X_test, y_test=y_test)
    print("Welcome to Takluban Language Identifier! Enter your sentences below:")
    # Define a set of supported languages
    supported_languages = {'tagalog', 'bikol', 'cebuano'}
    
    # Load the SVM model for profanity detection
    svm_model_path = "trained_profane_model.pkl"
    svm_model = joblib.load(svm_model_path)
    
    while True:
        sentence = input("Enter a sentence (or type 'exit' to quit): ").strip()
        if sentence.lower() == 'exit':
            print("Exiting the program.")
            break
        predicted_language = language_identifier.predict_language(sentence)
        # Check if the predicted language is supported
        if predicted_language not in supported_languages:
            print(f"Detected language: {predicted_language}. This language is not supported.")
            print("Skipping POS tagging and censorship for this sentence.\n")
            continue  # Skip to the next iteration if the language is not supported
        pattern_generator = get_pattern_generator(predicted_language)
        if pattern_generator:
            print(f"\nDetected language: {predicted_language}")
            censored_sentence, is_profane = predict_and_censor(sentence, pattern_generator, svm_model, predicted_language)
            if is_profane:
                print(f"Censored Sentence: {censored_sentence}")
            else:
                print(f"Cleaned Sentence: {censored_sentence}")
            save_to_csv(predicted_language, sentence, ' '.join(pattern_generator.tag_sentence(sentence)), censored_sentence)
            # Asking the user for the true label (1 = Profane, 0 = Not Profane)
            true_label = int(input("Is the sentence profane? (1 for profane, 0 for not profane): "))
            predictions.append(1 if is_profane else 0)
            true_labels.append(true_label)
            print(f"Sentence '{sentence}' saved with the detected language, POS tagging result, and censored sentence.\n")
        else:
            print(f"Pattern generator for {predicted_language} is not available.")
    
    # Confusion matrix and performance metrics calculation
    if len(predictions) > 0:
        print("Confusion Matrix and Performance Metrics:")
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
        print(f"Confusion Matrix:\n{cm}")
        
        # Check the shape of the confusion matrix
        if cm.shape == (2, 2):
            # Extract values from the confusion matrix
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle cases with fewer than 4 values
            tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
            fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
            fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
            tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        
        # Calculate Precision, Recall, F-Measure (F1 Score), and Accuracy
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Print the results
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        
        # Plot the confusion matrix
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Not Profane', 'Profane'], yticklabels=['Not Profane', 'Profane'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


if __name__ == "__main__":
    main()

#This shit is getting on my nerves