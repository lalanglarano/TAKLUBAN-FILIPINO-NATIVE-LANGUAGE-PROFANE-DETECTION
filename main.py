import csv
import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
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
        writer.writerow(['Language', 'Profane Sentence', 'POS Pattern'])  # Header for CSV

# Initialize lists to store predictions and true labels
predictions = []
true_labels = []

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
    path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar'

    if language == 'tagalog':
        return TagalogPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    elif language == 'bikol':
        return BikolPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    elif language == 'cebuano': 
        return CebuanoPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    else:
        print(f"No pattern generator available for {language}")
        return None

def predict_and_censor(sentence, pattern_generator, best_model, language, threshold=0.5):
    """Perform profanity detection and censorship using SVM and the pattern generator."""
    probas = best_model.predict_proba([sentence])[0]  # Predict probabilities using the SVM model
    
    is_profane = probas[1] >= threshold  # Only classify as profane if probability is above the threshold
    print(f"SVM Prediction: {'Profane' if is_profane else 'Not Profane'}")  # Print 'Profane' or 'Not Profane'

    pos_tagged_sentence = pattern_generator.tag_sentence(sentence)
    print(f"POS Tagged Sentence: {pos_tagged_sentence}")

    # If SVM says the sentence is profane, censor it
    if is_profane:
        censored_sentence = ' '.join(['*' * len(word) for word in sentence.split()])
        
        # Save the POS tag pattern to the profanity dictionary
        pos_pattern = ' '.join([item.split('|')[-1] for item in pos_tagged_sentence if '|' in item])
        save_profane_pattern(language, sentence, pos_pattern)
        
        print(f"Censoring the sentence. POS Pattern saved to profanity dictionary.")
        return censored_sentence, True  # Return censored sentence and True to indicate it's profane
    
    return sentence, False  # Return the original sentence and False to indicate it's not profane

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

            model_path = f'../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/{predicted_language}_trained_profane_model.pkl'
            print(f"Model path: {model_path}")
            if os.path.exists(model_path):
                best_model = joblib.load(model_path)
                print(f"Loaded SVM model for {predicted_language}.")
                
                censored_sentence, is_profane = predict_and_censor(sentence, pattern_generator, best_model, predicted_language)
                if is_profane:
                    print(f"Censored Sentence: {censored_sentence}")
                else:
                    print(f"Cleaned Sentence: {censored_sentence}")
            else:
                print(f"No SVM model found for {predicted_language}. Skipping censorship.")
                censored_sentence = sentence
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
