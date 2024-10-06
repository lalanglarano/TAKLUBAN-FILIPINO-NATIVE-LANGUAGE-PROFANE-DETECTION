import csv
import os
import joblib  # Add this for loading the model
from LanguageIdentification.FNLI import LanguageIdentification, ModelTraining  # Import necessary classes

# Define the path to save the results
output_file = "data.csv"

# Check if the CSV file already exists. If not, create it and add a header
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sentence', 'Language'])  # Header for CSV

def save_to_csv(sentence, language):
    """Save the sentence and detected language to a CSV file."""
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sentence, language])

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

        # Save the sentence and predicted language to the CSV file
        save_to_csv(sentence, predicted_language)

        print(f"Detected language: {predicted_language}")
        print(f"Sentence '{sentence}' saved with the detected language: {predicted_language}\n")

if __name__ == "__main__":
    main()
