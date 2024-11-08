import csv
import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import re
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
    "unta", "nga", "sa", "kani", "mo", "lang", "sila", "unsa", "ako", "niyo", "su"
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

# Additional function to print POS-tagged sentences
def print_pos_tagged_sentence(language, sentence, pos_tagged):
    """Print the language, sentence, and its POS-tagged result."""
    print(f"Language: {language}")
    print(f"Original Sentence: {sentence}")
    print(f"POS-Tagged Sentence: {pos_tagged}\n")

# Update in main process where POS tagging happens
def main():
    model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"
    dictionary_dir = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary"
    
    model, X_test, y_test = train_model_if_not_exists(model_path, dictionary_dir)
    language_identifier = LanguageIdentification(model=model, X_test=X_test, y_test=y_test)
    print("Welcome to Takluban Language Identifier! Enter your sentences below:")
    
    supported_languages = {'tagalog', 'bikol', 'cebuano'}
    svm_model_path = "trained_profane_model.pkl"
    svm_model = joblib.load(svm_model_path)
    
    while True:
        sentence = input("Enter a sentence (or type 'exit' to quit): ").strip()
        if sentence.lower() == 'exit':
            print("Exiting the program.")
            break
        
        predicted_language = language_identifier.predict_language(sentence)
        if predicted_language not in supported_languages:
            print(f"Detected language: {predicted_language}. This language is not supported.")
            continue
        
        pattern_generator = get_pattern_generator(predicted_language)
        if pattern_generator:
            pos_tagged = ' '.join(pattern_generator.tag_sentence(sentence))
            print_pos_tagged_sentence(predicted_language, sentence, pos_tagged)  # Print POS tagging result
            
            censored_sentence, is_profane = predict_and_censor(sentence, pattern_generator, svm_model, predicted_language)
            if is_profane:
                print(f"Censored Sentence: {censored_sentence}")
            else:
                print(f"Cleaned Sentence: {censored_sentence}")
            
            save_to_csv(predicted_language, sentence, pos_tagged, censored_sentence)
            true_label = int(input("Is the sentence profane? (1 for profane, 0 for not profane): "))
            predictions.append(1 if is_profane else 0)
            true_labels.append(true_label)
            print(f"Sentence '{sentence}' saved with the detected language, POS tagging result, and censored sentence.\n")
        else:
            print(f"Pattern generator for {predicted_language} is not available.")
    
    if len(predictions) > 0:
        cm = confusion_matrix(true_labels, predictions)
        print(f"Confusion Matrix:\n{cm}")
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Not Profane', 'Profane'], yticklabels=['Not Profane', 'Profane'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

if __name__ == "__main__":
    main()