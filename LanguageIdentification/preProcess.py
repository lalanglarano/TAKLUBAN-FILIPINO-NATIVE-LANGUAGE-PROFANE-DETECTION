import os
import csv
import re

class TextPreprocessor:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/dataset/dataset_{language}.csv"
        self.output_dir = f"{results_folder}/preprocessed/"
        self.output_file = f"{self.output_dir}/preprocessed_{language}.csv"
        self.dictionary_dir = f"{base_path}/LanguageIdentification/Dictionary/"
        self.dictionary_file = f"{self.dictionary_dir}/{language}_dictionary.csv"
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dictionary_dir, exist_ok=True)

    def preprocess_text(self, text):
        # Remove special characters, digits, and alphanumeric
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)

        # Convert to lowercase
        text = text.lower()

        return text

    def split_into_sentences(self, text):
        # Split the text based on punctuation marks: periods, exclamation marks, question marks, colons, and commas
        chunks = re.split(r'[.!?,]', text)

        sentences = []
        for chunk in chunks:
            # Split the chunk into words and apply the condition
            words = chunk.strip().split()
            if len(words) >= 4:  # Accept only if 4 or more consecutive words
                sentences.append(chunk.strip())

        return sentences

    def preprocess_csv(self):
        try:
            if not os.path.exists(self.input_file):
                print(f"Error: The file {self.input_file} does not exist.")
                return
            
            word_count = {}
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                lines = infile.readlines()
                writer = csv.writer(outfile)

                for line in lines:
                    preprocessed_line = self.preprocess_text(line)
                    sentences = self.split_into_sentences(preprocessed_line)

                    for sentence in sentences:
                        writer.writerow([sentence])

                        # Update the word count dictionary
                        words = sentence.split()
                        for word in words:
                            if not word.isnumeric():
                                word_count[word] = word_count.get(word, 0) + 1

            if word_count:
                with open(self.dictionary_file, 'w', newline='', encoding='utf-8') as dict_file:
                    writer = csv.writer(dict_file)
                    writer.writerow(['word', 'frequency'])
                    for word, freq in sorted(word_count.items()):
                        writer.writerow([word, freq])
                print(f"Dictionary saved at {self.dictionary_file}")
            else:
                print(f"No words found after preprocessing for {self.dictionary_file}.")

        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
#