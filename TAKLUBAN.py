import csv
import os
import pandas as pd
import joblib  # For loading the model
from LanguageIdentification.FNLI import LanguageIdentification, ModelTraining  # Import necessary classes
from POSTagging.POSTAGGER.pospkl.POSTagger import POSTagger  # Import the POS tagger from POS.py

# Define the path to save the results
output_file = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/POSdata.csv"

# Check if the CSV file already exists. If not, create it and add a header
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Language', 'Sentence', 'POS'])  # Header for CSV

def save_to_csv(language, sentence, pos_tagged):
    """Save the language, sentence, and POS tagged result to a CSV file."""
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([language, sentence, pos_tagged])

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

def perform_pos_tagging(language, sentence):
    """Perform POS tagging based on the identified language."""
    pos_tagger = POSTagger(language)  # Initialize the POS tagger for the given language

    # Show the POS tagging process in the output
    print(f"\nPerforming POS tagging for '{language}' on the sentence: {sentence}")
    
    # Perform POS tagging directly on the sentence and return the result
    pos_tagged = pos_tagger.pos_tag_text(sentence)
    
    return pos_tagged

def main():
    # Define the paths
    model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"
    dictionary_dir = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary"

    # Check if the model exists, if not train it
    model, X_test, y_test = train_model_if_not_exists(model_path, dictionary_dir)

    # Initialize the LanguageIdentification class with the loaded or newly trained model
    language_identifier = LanguageIdentification(model=model, X_test=X_test, y_test=y_test)

    print("Welcome to Takluban Language Identifier! Enter your sentences below:")

    while True:
        sentence = input("Enter a sentence (or type 'exit' to quit): ").strip()

        if sentence.lower() == 'exit':
            print("Exiting the program.")
            break

        # Identify the language of the sentence
        predicted_language = language_identifier.predict_language(sentence)

        # Perform POS tagging based on the identified language
        if predicted_language in ['cebuano', 'bikol', 'tagalog']:
            pos_tagged_sentence = perform_pos_tagging(predicted_language, sentence)

            # Save the result to the CSV file
            save_to_csv(predicted_language, sentence, pos_tagged_sentence)

            print(f"Detected language: {predicted_language}")
            print(f"POS Tagged Sentence: {pos_tagged_sentence}")
            print(f"Sentence '{sentence}' saved with the detected language and POS tagging result.\n")
        else:
            print(f"Unsupported language detected: {predicted_language}. No POS tagging performed.")

if __name__ == "__main__":
    main()
