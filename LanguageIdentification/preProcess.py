"""
Program Title: Text Preprocessor

Programmers: Jeo Abarre, Annalyn Belen, Telish Gonzales, Randolph Larano

Where the program fits in the general system designs: This module preprocesses text data for 
language identification and profanity detection.

Date written and last revised: September 26, 2024   |   November 18, 2024

Purpose: To clean and preprocess text data, splitting it into sentences and removing noise. 
This module reads raw text data from a CSV file, preprocesses the text by removing numbers and 
special characters, splits the text into sentences, and saves the preprocessed sentences to a new CSV file.

Data structures, algorithms, and control:

Data Structures:
self.input_file: Path to the input CSV file containing raw text data.
self.output_file: Path to the output CSV file for saving preprocessed text data.

Algorithms:
Uses regular expressions to clean and preprocess text.
Splits text into sentences based on punctuation.
Writes preprocessed sentences to a CSV file.

Control:
Initializes with paths to input and output files.
Preprocesses text by removing numbers and special characters.
Splits text into sentences.
Saves preprocessed sentences to a CSV file.
"""

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
        """Clean and lowercase text, removing numbers, special characters, and extra whitespace."""
        # Remove all numbers
        text = re.sub(r'\d+', '', text)
        # Remove specified special characters and retain only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
        # Normalize whitespace by collapsing multiple spaces into a single space
        return re.sub(r'\s+', ' ', text)

    @staticmethod
    def split_into_sentences(text):
        """Split text into sentences of 4 or more words, ignoring empty sentences."""
        return [chunk.strip() for chunk in re.split(r'[.!?]', text) if len(chunk.split()) >= 4 and chunk.strip()]

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
                    profane = row[1]

                    # Preprocess the sentence
                    preprocessed_sentence = self.preprocess_text(sentence)

                    # Split into multiple sentences if needed
                    sentences = self.split_into_sentences(preprocessed_sentence)

                    # Write each sentence with the corresponding label
                    for processed_sentence in sentences:
                        writer.writerow([processed_sentence, profane])
        
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    languages = ['bikol', 'tagalog', 'cebuano']
    for language in languages:
        processor = TextPreprocessor(language)
        processor.process_file()
