import os
import re
import csv

class TextPreprocessor:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        self.input_file = os.path.join(base_path, f"Results/dataset/dataset_{language}.csv")
        self.output_file = os.path.join(base_path, f"Results/PFW/preprocessed_{language}.csv")

        os.makedirs(os.path.dirname(self.input_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    @staticmethod
    def preprocess_text(text):
        """Clean and lowercase text, removing special characters, numbers, and quotes."""
        # Remove numbers and quotes, retain letters and spaces only
        return re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()

    @staticmethod
    def split_into_sentences(text):
        """Split text into sentences of 4 or more words, ignoring empty sentences."""
        return [chunk.strip() for chunk in re.split(r'[.!?]', text) if len(chunk.split()) >= 4 and chunk.strip()]

    def process_file(self):
        """Preprocess text and save sentences to a CSV file, including already-preprocessed sentences."""
        if not os.path.exists(self.input_file):
            print(f"Error: The file {self.input_file} does not exist.")
            return

        try:
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                # Write header to output file
                writer.writerow(["sentence"])

                # Process each row
                for row in reader:
                    sentence = row[0].strip()

                    # Preprocess the sentence if needed
                    preprocessed_sentence = self.preprocess_text(sentence)

                    # Split into sentences, ensuring each has 4+ words
                    sentences = self.split_into_sentences(preprocessed_sentence)

                    # If no splitting occurred, retain the original (single) preprocessed sentence
                    if not sentences:
                        sentences = [preprocessed_sentence]

                    # Write each sentence to the output file
                    for processed_sentence in sentences:
                        writer.writerow([processed_sentence])

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    languages = ['bikol', 'tagalog', 'cebuano']
    for language in languages:
        processor = TextPreprocessor(language)
        processor.process_file()
