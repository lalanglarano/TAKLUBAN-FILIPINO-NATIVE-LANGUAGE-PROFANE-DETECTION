import csv
from LanguageIdentification.FNLI import LanguageIdentification  # Adjust the path if necessary

def identify_language(sentence, language_identifier):
    """Identify the language of the input sentence using the LanguageIdentification class."""
    try:
        return language_identifier.predict_language(sentence)
    except Exception as e:
        print(f"Error identifying language: {e}")
        return "Unknown"

def main():
    # Initialize the LanguageIdentification class
    dictionary_dir = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary"
    language_identifier = LanguageIdentification(dictionary_dir)

    # Ask the user to input a sentence
    sentence = input("Please enter a sentence: ")

    # Identify the language using the LanguageIdentification class
    language = identify_language(sentence, language_identifier)

    # Save the sentence and its identified language to data.csv
    try:
        with open('data.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([sentence, language])

        print(f"Sentence: '{sentence}' has been identified as: '{language}' and saved to data.csv")
    except Exception as e:
        print(f"Error saving to file: {e}")

if __name__ == "__main__":
    main()
