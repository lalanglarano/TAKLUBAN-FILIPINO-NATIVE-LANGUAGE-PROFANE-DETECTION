import os
import pandas as pd
import re
from collections import Counter

class LanguageIdentification:
    def __init__(self, dictionary_dir):
        self.dictionary_dir = dictionary_dir
        self.noise_words = self.initialize_noise_words()
        self.word_dictionaries = self.load_dictionaries()

    def initialize_noise_words(self):
        """Initialize common noise words for Tagalog, Bikol, and Cebuano."""
        return {
            'Tagalog': {"na", "nang", "ng", "mga", "ang", "kung", "yan", "ito", "si", "ko", "po"},
            'Bikol': {"tabi", "ngani", "ini", "kang", "iyo", "hali", "baga", "ho", "mo", "ba", "si"},
            'Cebuano': {"dayon", "gani", "kana", "mao", "pud", "bitaw", "ta", "si", "ug"}
        }

    def load_dictionaries(self):
        """Load word dictionaries for all languages."""
        word_sets = {}
        for language in ['Tagalog', 'Bikol', 'Cebuano']:
            dict_file = f"{self.dictionary_dir}/{language.lower()}_dictionary.csv"
            if os.path.exists(dict_file):
                # Load words directly from the CSV (assumes single-column CSV with words)
                df = pd.read_csv(dict_file, usecols=[0], header=None, names=['word'])
                # Convert word column into a set for quick lookup
                word_sets[language] = set(df['word'].dropna().str.lower())  # Ensure words are lowercase
            else:
                print(f"Warning: Dictionary file {dict_file} not found.")
        return word_sets

    def remove_noise(self, words, language):
        """Remove noise words from the list of words."""
        return [word for word in words if word not in self.noise_words[language]]

    def predict_language(self, sentence):
        """Predict the language of a sentence based on word existence in dictionaries."""
        words = sentence.split()
        scores = {lang: 0 for lang in self.word_dictionaries}

        for lang, word_set in self.word_dictionaries.items():
            cleaned_words = self.remove_noise(words, lang)
            for word in cleaned_words:
                if word in word_set:  # Check if word exists in the dictionary
                    scores[lang] += 1  # Increment score for the matching language

        return scores

    def determine_language(self, sentences):
        """Determine the dominant language from a list of sentences."""
        language_counter = Counter()

        for sentence in sentences:
            scores = self.predict_language(sentence)
            dominant_language = max(scores, key=scores.get)  # Get language with the highest score
            language_counter[dominant_language] += 1

        return language_counter.most_common(1)[0][0] if language_counter else None


if __name__ == "__main__":
    # Sample test for language identification
    sentences = ["Magayunon kang maray masiramon kang padabaon"]
    language_identifier = LanguageIdentification(f"../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/RuleGeneration_LanguageIdentification/Dictionary")
    dominant_language = language_identifier.determine_language(sentences)
    print(f"The dominant language is: {dominant_language}")
