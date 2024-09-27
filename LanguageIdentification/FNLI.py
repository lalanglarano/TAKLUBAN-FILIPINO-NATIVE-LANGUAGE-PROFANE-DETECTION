import os
import numpy as np
import pandas as pd
import re
import csv
from collections import Counter

# Preprocessing class for basic text cleaning and sentence splitting
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

        # Define noise words specific to the language
        self.noise_words = {
            'tagalog': set(["na", "nang", "ng", "mga", "ang", "kung", "yan", "yun", "ayan", "sina",
                            "baka", "ano", "anong", "mag", "doon", "mo", "so", "po", "ko", "eme", "may", 
                            "luh", "ito", "ay", "ganon", "lang", "dito", "pang", "daw", "raw", "si"]),
            'bikol': set(["nem", "ngani", "tabi", "ning", "kamo", "ini", "iyo", "hali", "bala", "aba", 
                          "alin", "baga", "ganiyan", "gaya", "ho", "ika", "kay", "mo", "naman", "wag", 
                          "naman", "yata", "ba", "si", "garo", "ho"]),
            'cebuano': set(["dayon", "gani", "kana", "mao", "diay", "mao ni", "mao ba", "lang", "usa", "kang",
                            "kita", "kita tanan", "kamo", "ta", "gyud", "bitaw", "pud", "kay", "ahh", "sa", "si",
                            "pag", "pwede", "pwes", "pano", "ug"])
        }[language]

    def preprocess_text(self, text):
        # Convert to lowercase and remove non-alphanumeric characters except spaces and commas
        text = text.lower()
        text = ''.join(char if char.isalnum() or char in [' ', ','] else '' for char in text)
        return text

    def split_into_sentences(self, text):
        # Split the text based on punctuation marks
        chunks = re.split(r'[.!?]', text)
        sentences = [chunk.strip() for chunk in chunks if len(chunk.split()) >= 2]
        return sentences

    def remove_noise_words(self, words):
        # Remove noise words from a list of words
        return [word for word in words if word not in self.noise_words]

    def preprocess_csv(self):
        word_count = {}
        try:
            if not os.path.exists(self.input_file):
                print(f"Error: The file {self.input_file} does not exist.")
                return
            
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                lines = infile.readlines()
                writer = csv.writer(outfile)

                for line in lines:
                    preprocessed_line = self.preprocess_text(line)
                    sentences = self.split_into_sentences(preprocessed_line)

                    for sentence in sentences:
                        words = sentence.split()
                        cleaned_words = self.remove_noise_words(words)  # Remove noise words
                        cleaned_sentence = ' '.join(cleaned_words)

                        # Write cleaned sentence to the output file
                        if cleaned_sentence:  # Only write non-empty sentences
                            writer.writerow([cleaned_sentence])

                        # Update the word count dictionary
                        for word in cleaned_words:
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
                print(f"No words found after preprocessing for {self.dictionary_file}")

        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Class for handling language identification with noise word removal
class LanguageIdentification:
    def __init__(self, data, dictionary_dir):
        self.data = data
        self.dictionary_dir = dictionary_dir
        
        # Define noise words for each language
        self.noise_words = {
            'Tagalog': set(["na", "nang", "ng", "mga", "ang", "kung", "yan", "yun", "ayan", "sina",
                            "baka", "ano", "anong", "mag", "doon", "mo", "so", "po", "ko", "eme", "may", 
                            "luh", "ito", "ay", "ganon", "lang", "dito", "pang", "daw", "raw", "si"]),
            'Bikol': set(["nem", "ngani", "tabi", "ning", "kamo", "ini", "iyo", "hali", "bala", "aba", 
                          "alin", "baga", "ganiyan", "gaya", "ho", "ika", "kay", "mo", "naman", "wag", 
                          "naman", "yata", "ba", "si", "garo", "ho"]),
            'Cebuano': set(["dayon", "gani", "kana", "mao", "diay", "mao ni", "mao ba", "lang", "usa", "kang"
                            "kita", "kita tanan", "kamo", "ta", "gyud", "bitaw", "pud", "kay", "ahh", "sa", "si"
                            "pag", "pwede", "pwes", "pano", "ug"])
        }

        # Load dictionaries
        self.word_frequencies = {}
        self.load_dictionaries()

    def load_dictionaries(self):
        for language in ['Tagalog', 'Bikol', 'Cebuano']:
            dict_file = f"{self.dictionary_dir}/{language.lower()}_dictionary.csv"
            if os.path.exists(dict_file):
                df = pd.read_csv(dict_file)
                self.word_frequencies[language] = dict(zip(df['word'], df['frequency']))
            else:
                print(f"Warning: Dictionary file {dict_file} does not exist.")

    def remove_noise_words(self, words, language):
        noise_words_for_language = self.noise_words.get(language, set())
        cleaned_words = [word for word in words if word not in noise_words_for_language]
        return cleaned_words

    def predict_language(self, sentence):
        # Clean the sentence
        words = sentence.split()
        words = self.remove_noise_words(words, 'Tagalog') + self.remove_noise_words(words, 'Bikol') + self.remove_noise_words(words, 'Cebuano')

        language_scores = {language: 0 for language in self.word_frequencies.keys()}

        # Calculate scores based on word frequencies
        for word in words:
            for language, frequency_dict in self.word_frequencies.items():
                if word in frequency_dict:
                    language_scores[language] += frequency_dict[word]

        return language_scores

    def get_dominant_language(self, sentences):
        # Initialize a counter for languages
        language_counter = Counter()

        for sentence in sentences:
            scores = self.predict_language(sentence)
            # Find the language with the highest score for the current sentence
            predicted_language = max(scores, key=scores.get)
            language_counter[predicted_language] += 1

        # Determine the language that is dominant based on counts
        dominant_language = language_counter.most_common(1)[0][0] if language_counter else None
        return dominant_language

# Main flow for preprocessing and language identification
if __name__ == "__main__":
    languages = ['tagalog', 'bikol', 'cebuano']

    # Preprocess files for Tagalog, Bikol, and Cebuano
    for language in languages:
        processor = TextPreprocessor(language)
        processor.preprocess_csv()

    # Load the preprocessed datasets
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/preprocessed/"
    dataframes = []

    for language in languages:
        df = pd.read_csv(f"{base_path}/preprocessed_{language}.csv", header=None, names=['sentence'])
        dataframes.append(df)

    # Concatenate all dataframes into one
    all_data = pd.concat(dataframes, ignore_index=True)

    # Initialize language identifier
    language_identifier = LanguageIdentification(all_data, f"../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary/")

    # Example sentences to predict the dominant language
    sample_sentences = ["ambot sa imong tanan"]
    dominant_language = language_identifier.get_dominant_language(sample_sentences)
    print(f"The dominant language is: {dominant_language}")