import os
import re
import csv

class TextPreprocessor:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/dataset/dataset_{language}.csv"
        self.output_file = f"{results_folder}/preprocessed/preprocessed_{language}.csv"

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def preprocess_text(self, text):
        """Clean and lowercase text, removing special characters."""
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        return text

    def split_into_sentences(self, text):
        """Split text into sentences of 4 or more words."""
        return [chunk.strip() for chunk in re.split(r'[.!?]', text) if len(chunk.split()) >= 4]

    def process_file(self):
        """Preprocess text and save sentences to a CSV file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                lines = infile.readlines()
                writer = csv.writer(outfile)

                for line in lines:
                    preprocessed_line = self.preprocess_text(line)
                    sentences = self.split_into_sentences(preprocessed_line)

                    for sentence in sentences:
                        writer.writerow([sentence])
        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    languages = ['tagalog', 'bikol', 'cebuano']
    for language in languages:
        processor = TextPreprocessor(language)
        processor.process_file()
