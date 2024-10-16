import csv
import os
import pandas as pd
import joblib
from LanguageIdentification.FNLI import LanguageIdentification, ModelTraining
from POSTagging.POSTAGGER.pospkl.POSTagger import POSTagger
from PATTERN_GENERATION.tagalog import PatternGenerator as TagalogPatternGenerator  # Import PatternGenerator for Tagalog
from PATTERN_GENERATION.bikol import PatternGenerator as BikolPatternGenerator  # Import PatternGenerator for Bikol
from PATTERN_GENERATION.cebuano import PatternGenerator as CebuanoPatternGenerator 

# Define the path to save the results
output_file = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/POSdata.csv"

# Check if the CSV file already exists. If not, create it and add a header
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Language', 'Sentence', 'POS', 'Censored Sentence'])  # Header for CSV

def save_to_csv(language, sentence, pos_tagged, censored_sentence):
    """Save the language, sentence, POS tagged result, and censored sentence to a CSV file."""
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([language, sentence, pos_tagged, censored_sentence])

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
    """Return the appropriate pattern generator for the given language."""
    predefined_rules_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/PATTERN_GENERATION/predefined_rules.csv"
    model_filename = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar'

    if language == 'tagalog':
        return TagalogPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    elif language == 'bikol':
        return BikolPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    elif language == 'cebuano':  # Add Cebuano pattern generator here
        return CebuanoPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    else:
        return None

def predict_and_censor(sentence, pattern_generator, best_model, threshold=0.9):
    """Perform profanity detection and censorship using SVM and the pattern generator."""
    # Step 1: Use SVM to predict if the sentence is profane
    probas = best_model.predict_proba([sentence])[0]  # Predict probabilities using the SVM model
    print(f"Probabilities: {probas}")
    
    is_profane = probas[1] >= threshold  # Only classify as profane if probability is above the threshold
    print(f"SVM Prediction - {'Profane' if is_profane else 'Not Profane'} with probability: {probas[1]}")

    # If SVM says the sentence is profane, censor the entire sentence
    if is_profane:
        print(f"Censoring the entire sentence due to SVM detection.")
        return '*****'  # Censor the entire sentence
    
    # Step 2: If the sentence is not detected as profane, return it as-is
    return sentence  # Return the sentence uncensored if not profane

def main():
    # Define the paths
    model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"
    dictionary_dir = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary"

    # Train the language identification model if it doesn't exist
    model, X_test, y_test = train_model_if_not_exists(model_path, dictionary_dir)

    # Initialize the LanguageIdentification class with the loaded model
    language_identifier = LanguageIdentification(model=model, X_test=X_test, y_test=y_test)

    print("Welcome to Takluban Language Identifier! Enter your sentences below:")

    while True:
        sentence = input("Enter a sentence (or type 'exit' to quit): ").strip()

        if sentence.lower() == 'exit':
            print("Exiting the program.")
            break

        # Step 1: Identify the language of the sentence
        predicted_language = language_identifier.predict_language(sentence)

        # Step 2: Initialize PatternGenerator for detected language
        pattern_generator = get_pattern_generator(predicted_language)

        if pattern_generator and predicted_language in ['cebuano', 'bikol', 'tagalog']:
            print(f"\nDetected language: {predicted_language}")
            
            # Step 3: Perform POS tagging
            pos_tagged_sentence = pattern_generator.tag_sentence(sentence)
            print(f"POS Tagged Sentence: {pos_tagged_sentence}")

            # Step 4: Load the SVM model for profanity detection
            model_path = f'../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/{predicted_language}_trained_profane_model.pkl'
            if os.path.exists(model_path):
                best_model = joblib.load(model_path)
                print(f"Loaded SVM model for {predicted_language}.")
                
                # Step 5: Perform profanity detection and censoring
                censored_sentence = predict_and_censor(sentence, pattern_generator, best_model)
                print(f"Censored Sentence: {censored_sentence}")
            else:
                print(f"No SVM model found for {predicted_language}. Skipping censorship.")
                censored_sentence = sentence  # No model available, skip censoring

            # Step 6: Save the result to the CSV file
            save_to_csv(predicted_language, sentence, pos_tagged_sentence, censored_sentence)

            print(f"Sentence '{sentence}' saved with the detected language, POS tagging result, and censored sentence.\n")
        else:
            print(f"Unsupported language detected: {predicted_language}. No POS tagging performed.")

if __name__ == "__main__":
    main()
