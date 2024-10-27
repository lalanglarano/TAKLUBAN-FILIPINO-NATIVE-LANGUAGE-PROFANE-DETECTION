import os
import re
import csv

class TextPreprocessor:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        self.input_file = os.path.join(base_path, f"UsedDataset/dataset_{language}.csv")
        self.output_file = os.path.join(base_path, f"Results/PFW/preprocessed_{language}.csv")

        os.makedirs(os.path.dirname(self.input_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    @staticmethod
    def preprocess_text(text):
        """Clean and lowercase text, removing special characters."""
        return re.sub(r'[^a-zA-Z\s]', '', text).lower()

    @staticmethod
    def split_into_sentences(text):
        """Split text into sentences of 4 or more words."""
        return [chunk.strip() for chunk in re.split(r'[.!?]', text) if len(chunk.split()) >= 4]

    def process_file(self):
        """Preprocess text and save sentences with labels to a CSV file."""
        if not os.path.exists(self.input_file):
            print(f"Error: The file {self.input_file} does not exist.")
            return

        try:
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                header = next(reader) 
                writer.writerow(header)

                # Process each row
                for row in reader:
                    sentence = row[0]
                    label = row[1]

                    # Preprocess the sentence
                    preprocessed_sentence = self.preprocess_text(sentence)

                    # Split into multiple sentences if needed
                    sentences = self.split_into_sentences(preprocessed_sentence)

                    # Write each sentence with the corresponding label
                    for processed_sentence in sentences:
                        writer.writerow([processed_sentence, label])
        
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    languages = ['bikol', 'tagalog', 'cebuano']
    for language in languages:
        processor = TextPreprocessor(language)
        processor.process_file()
