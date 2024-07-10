import csv
import re
import string

class TextPreprocessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    @staticmethod
    def preprocess_text(text):
        # Convert text to lowercase
        text = text.lower()
        # Handle words with apostrophes
        text = re.sub(r"(\w)'(\w)", r"\1\2", text)
        # Remove punctuation and special characters except numbers
        text = ''.join(char for char in text if char.isalnum() or char == ' ')
        # Remove isolated numbers (numbers surrounded by whitespace)
        text = re.sub(r'\b\d+\b', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def preprocess_csv(self):
        try:
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                
                # Process each row (assuming each row is a line of text)
                for row in reader:
                    preprocessed_row = [self.preprocess_text(cell) for cell in row]
                    writer.writerow(preprocessed_row)
        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

# File paths
tagalog_input_file = 'dataset_tagalog.csv'
bikol_input_file = 'dataset_bikol.csv'
cebuano_input_file = 'dataset_cebuano.csv'

tagalog_output_file = 'preprocessed_tagalog.csv'
bikol_output_file = 'preprocessed_bikol.csv'
cebuano_output_file = 'preprocessed_cebuano.csv'

# Create instances of TextPreprocessor for each language
tagalog_preprocessor = TextPreprocessor(tagalog_input_file, tagalog_output_file)
bikol_preprocessor = TextPreprocessor(bikol_input_file, bikol_output_file)
cebuano_preprocessor = TextPreprocessor(cebuano_input_file, cebuano_output_file)

# Preprocess the CSV files
tagalog_preprocessor.preprocess_csv()
bikol_preprocessor.preprocess_csv()
cebuano_preprocessor.preprocess_csv()
