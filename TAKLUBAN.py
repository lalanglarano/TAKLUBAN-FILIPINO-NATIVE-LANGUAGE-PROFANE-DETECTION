import csv
import os
import joblib  # Add this for loading the model
from LanguageIdentification.FNLI import LanguageIdentification  # Import only the required class

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

def main():
    # Load the pre-trained model from the saved file
    model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"
    
    # Use joblib to load the pre-trained model
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The model file {model_path} does not exist.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # Initialize the LanguageIdentification class with the loaded model
    language_identifier = LanguageIdentification(model=model, X_test=[], y_test=[])

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
