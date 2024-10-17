import csv
import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from LanguageIdentification.FNLI import LanguageIdentification, ModelTraining
from POSTagging.POSTAGGER.pospkl.POSTagger import POSTagger  # Import the POSTagger class from POSTagger.py

# Global paths to avoid redundancy
model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"
dictionary_dir = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary"
output_file = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/POSdata.csv"
profanity_dictionary_file = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/profanity_dictionary.csv"

# Function to check if the CSV file exists and create if necessary
def initialize_csv():
    """Ensure that the CSV file exists and has a header."""
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Language', 'Sentence', 'POS', 'Censored Sentence'])  # Header for CSV

# Centralized function for loading or training the language identification model
def load_or_train_model():
    """Load or train the language identification model."""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Training a new model...")
        trainer = ModelTraining(dictionary_dir)
        model, X_test, y_test = trainer.train_model()
        joblib.dump(model, model_path)
        print(f"Model trained and saved at {model_path}.")
    else:
        print(f"Loading pre-saved model from {model_path}.")
        model = joblib.load(model_path)
        X_test, y_test = [], []  # Empty test sets as they aren't needed for prediction
    return LanguageIdentification(model=model, X_test=X_test, y_test=y_test)

def get_pos_tagger(language):
    """Return the appropriate POS tagger from POSTagger.py for the given language."""
    if language in ['tagalog', 'bikol', 'cebuano']:
        return POSTagger(language)  # Create an instance of the POS tagger for the detected language
    return None

def predict_and_censor(sentence, best_model, threshold=0.5):
    """Perform profanity detection and censorship using SVM."""
    probas = best_model.predict_proba([sentence])[0]  # Predict probabilities using the SVM model
    
    is_profane = probas[1] >= threshold  # Only classify as profane if probability is above the threshold
    print(f"SVM Prediction: {'Profane' if is_profane else 'Not Profane'}")

    # If SVM says the sentence is profane, censor it
    if is_profane:
        print(f"Censoring the sentence based on its length.")
        censored_sentence = ' '.join(['*' * len(word) for word in sentence.split()])
        return censored_sentence, True  # Return censored sentence and True to indicate it's profane
    
    return sentence, False  # Return the original sentence and False to indicate it's not profane

# Central function for processing a sentence
def process_sentence(sentence, language_identifier):
    """Process the sentence to predict the language, POS tag it using POSTagger, apply regex rules, and censor if necessary."""
    predicted_language = language_identifier.predict_language(sentence)
    pos_tagger = get_pos_tagger(predicted_language)

    if pos_tagger and predicted_language in ['cebuano', 'bikol', 'tagalog']:
        # Perform initial POS tagging using Stanford POS Tagger
        pos_tagged_sentence = pos_tagger.pos_tag_text(sentence)

        profanity_model_path = f'../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/{predicted_language}_trained_profane_model.pkl'
        
        if os.path.exists(profanity_model_path):
            best_model = joblib.load(profanity_model_path)
            censored_sentence, is_profane = predict_and_censor(sentence, best_model)
            save_to_csv(predicted_language, sentence, pos_tagged_sentence, censored_sentence)

            # Save the POS tag to profanity_dictionary.csv only if the sentence is profane
            if is_profane:
                save_pos_to_profanity_dictionary(pos_tagged_sentence)
        else:
            censored_sentence = sentence
            is_profane = False

        return predicted_language, pos_tagged_sentence, censored_sentence, is_profane
    return "Unsupported language", None, sentence, False

def save_pos_to_profanity_dictionary(pos_tagged):
    """Save only the POS tag to the profanity dictionary CSV file."""
    with open(profanity_dictionary_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([pos_tagged])

# Original function if you still need it for other data
def save_to_csv(language, sentence, pos_tagged, censored_sentence):
    """Save the language, sentence, POS tagged result, and censored sentence to a CSV file."""
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([language, sentence, pos_tagged, censored_sentence])

# Main function to run the program
def main():
    # Initialize CSV file and load or train the model
    initialize_csv()
    language_identifier = load_or_train_model()

    print("Welcome to Takluban Language Identifier! Enter your sentences below:")
    
    predictions = []
    true_labels = []

    while True:
        sentence = input("Enter a sentence (or type 'exit' to quit): ").strip()

        if sentence.lower() == 'exit':
            print("Exiting the program.")
            break

        predicted_language, pos_tagged_sentence, censored_sentence, is_profane = process_sentence(sentence, language_identifier)

        if predicted_language in ['cebuano', 'bikol', 'tagalog']:
            print(f"Detected language: {predicted_language}")
            print(f"POS Tagged Sentence: {pos_tagged_sentence}")
            print(f"{'Censored Sentence' if is_profane else 'Cleaned Sentence'}: {censored_sentence}")

            save_to_csv(predicted_language, sentence, pos_tagged_sentence, censored_sentence)

            # Asking the user for the true label (1 = Profane, 0 = Not Profane)
            true_label = int(input("Is the sentence profane? (1 for profane, 0 for not profane): "))
            predictions.append(1 if is_profane else 0)
            true_labels.append(true_label)

            print(f"Sentence '{sentence}' saved with the detected language, POS tagging result, and censored sentence.\n")
        else:
            print(f"Unsupported language detected: {predicted_language}. No POS tagging performed.")
    
    # Confusion matrix and performance metrics calculation
    if len(predictions) > 0:
        print("Confusion Matrix and Performance Metrics:")
        cm = confusion_matrix(true_labels, predictions)
        print(f"Confusion Matrix:\n{cm}")
        
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        # Plot the confusion matrix
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Not Profane', 'Profane'], yticklabels=['Not Profane', 'Profane'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

if __name__ == "__main__":
    main()
