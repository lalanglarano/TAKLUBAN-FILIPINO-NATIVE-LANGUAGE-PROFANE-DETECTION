import os
import numpy as np
import pandas as pd
import re
import csv
from collections import Counter


# TextPreprocessor class for text cleaning and sentence splitting
class TextPreprocessor:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/dataset/dataset_{language}.csv"
        self.output_file = f"{results_folder}/preprocessed/preprocessed_{language}.csv"
        self.dictionary_file = f"{base_path}/LanguageIdentification/Dictionary/{language}_dictionary.csv"
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.dictionary_file), exist_ok=True)

    def preprocess_text(self, text):
        """Clean and lowercase text, removing special characters."""
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        return text

    def split_into_sentences(self, text):
        """Split text into sentences of 4 or more words."""
        return [chunk.strip() for chunk in re.split(r'[.!?]', text) if len(chunk.split()) >= 4]

    def process_file(self):
        """Preprocess text and create word frequency dictionary."""
        word_count = {}
        try:
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                lines = infile.readlines()
                writer = csv.writer(outfile)

                for line in lines:
                    preprocessed_line = self.preprocess_text(line)
                    sentences = self.split_into_sentences(preprocessed_line)

                    for sentence in sentences:
                        writer.writerow([sentence])
                        for word in sentence.split():
                            word_count[word] = word_count.get(word, 0) + 1

            return word_count
        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")


# LanguageIdentification class for removing noise words and predicting language
class LanguageIdentification:
    def __init__(self, dictionary_dir):
        self.dictionary_dir = dictionary_dir
        self.noise_words = self.initialize_noise_words()
        self.word_frequencies = self.load_dictionaries()

    def initialize_noise_words(self):
        """Initialize common noise words for Tagalog, Bikol, and Cebuano."""
        return {
            'Tagalog': {"na", "nang", "ng", "mga", "ang", "kung", "yan", "ito", "si", "ko", "po"},
            'Bikol': {"tabi", "ngani", "ini", "iyo", "hali", "baga", "ho", "mo", "ba", "si"},
            'Cebuano': {"dayon", "gani", "kana", "mao", "pud", "bitaw", "ta", "si", "ug"}
        }

    def load_dictionaries(self):
        """Load word frequency dictionaries for all languages."""
        frequencies = {}
        for language in ['Tagalog', 'Bikol', 'Cebuano']:
            dict_file = f"{self.dictionary_dir}/{language.lower()}_dictionary.csv"
            if os.path.exists(dict_file):
                df = pd.read_csv(dict_file)
                frequencies[language] = dict(zip(df['word'], df['frequency']))
            else:
                print(f"Warning: Dictionary file {dict_file} not found.")
        return frequencies

    def remove_noise(self, words, language):
        """Remove noise words from the list of words."""
        return [word for word in words if word not in self.noise_words[language]]

    def predict_language(self, sentence):
        """Predict the language of a sentence based on word frequencies."""
        words = sentence.split()
        scores = {lang: 0 for lang in self.word_frequencies}

        for lang, freq_dict in self.word_frequencies.items():
            cleaned_words = self.remove_noise(words, lang)
            for word in cleaned_words:
                scores[lang] += freq_dict.get(word, 0)

        return scores

    def determine_language(self, sentences):
        """Determine the dominant language from a list of sentences."""
        language_counter = Counter()

        for sentence in sentences:
            scores = self.predict_language(sentence)
            dominant_language = max(scores, key=scores.get)
            language_counter[dominant_language] += 1

        return language_counter.most_common(1)[0][0] if language_counter else None


if __name__ == "__main__":
    languages = ['tagalog', 'bikol', 'cebuano']

    # Preprocess and generate dictionaries for each language
    for language in languages:
        processor = TextPreprocessor(language)
        word_count = processor.process_file()

        # Remove noise words and save dictionary
        language_identifier = LanguageIdentification(f"../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary")
        filtered_word_count = {word: freq for word, freq in word_count.items() if word not in language_identifier.noise_words[language.capitalize()]}

        # Save the dictionary
        with open(f"../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary/{language}_dictionary.csv", 'w', newline='', encoding='utf-8') as dict_file:
            writer = csv.writer(dict_file)
            writer.writerow(['word', 'frequency'])
            for word, freq in sorted(filtered_word_count.items()):
                writer.writerow([word, freq])

    # Sample test for language identification
    sentences = ["maray"]
    language_identifier = LanguageIdentification(f"../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary")
    dominant_language = language_identifier.determine_language(sentences)
    print(f"The dominant language is: {dominant_language}")
