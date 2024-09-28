import os
import csv
import subprocess
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DictionaryGenerator:
    def __init__(self, preprocessed_dir, dictionary_dir):
        self.preprocessed_dir = preprocessed_dir
        self.dictionary_dir = dictionary_dir
        self.noise_words = self.initialize_noise_words()
        os.makedirs(self.dictionary_dir, exist_ok=True)

    def initialize_noise_words(self):
        """Initialize common noise words for Tagalog, Bikol, and Cebuano."""
        return {
            'Tagalog': {"na", "nang", "ng", "mga", "ang", "kung", "yan", "ito", "si", "ko", "po"},
            'Bikol': {"ta", "ngani", "ini", "kang", "iyo", "hali", "baga", "ho", "mo", "ba", "si"},
            'Cebuano': {"dayon", "gani", "kana", "mao", "pud", "bitaw", "ta", "si", "ug"}
        }

    def remove_noise(self, words, language):
        """Remove noise words from the list of words."""
        return [word for word in words if word.lower() not in self.noise_words[language.capitalize()]]

    def generate_dictionary(self, language):
        """Generate a word frequency dictionary from preprocessed sentences."""
        word_count = Counter()
        preprocessed_file = os.path.join(self.preprocessed_dir, f"preprocessed_{language}.csv")

        try:
            with open(preprocessed_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                for row in reader:
                    if row:  # Check if the row is not empty
                        sentence = row[0]
                        words = sentence.split()
                        # Remove noise words before counting
                        cleaned_words = self.remove_noise(words, language)
                        word_count.update(cleaned_words)

            # Save the dictionary
            self.save_dictionary(word_count, language)

        except FileNotFoundError:
            print(f"Error: The file {preprocessed_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def save_dictionary(self, word_count, language):
        """Save the word frequency dictionary to a CSV file."""
        dict_file = os.path.join(self.dictionary_dir, f"{language}_dictionary.csv")
        with open(dict_file, 'w', newline='', encoding='utf-8') as dict_file:
            writer = csv.writer(dict_file)
            writer.writerow(['word', 'frequency'])
            for word, freq in sorted(word_count.items()):
                writer.writerow([word, freq])
        print(f"Dictionary saved at {dict_file}")

class LanguageIdentification:
    def __init__(self, dictionary_dir):
        self.dictionary_dir = dictionary_dir
        self.word_frequencies = self.load_dictionaries()
        self.model = self.train_model()

    def load_dictionaries(self):
        frequencies = {}
        for language in ['tagalog', 'bikol', 'cebuano']:
            dict_file = os.path.join(self.dictionary_dir, f"{language}_dictionary.csv")
            if os.path.exists(dict_file):
                df = pd.read_csv(dict_file)
                frequencies[language] = dict(zip(df['word'], df['frequency']))
                print(f"Loaded {language} dictionary with {len(frequencies[language])} entries.")
                # Check if the dictionary is empty
                if not frequencies[language]:
                    print(f"Warning: The dictionary for {language} is empty.")
            else:
                print(f"Dictionary file for {language} not found.")
        return frequencies
    
    def train_model(self):
        # Prepare training data
        data = []
        labels = []
        for language, word_freq in self.word_frequencies.items():
            for word, freq in word_freq.items():
                if freq > 0 and isinstance(word, str):  # Check for positive frequency and valid word
                    data.extend([word] * freq)
                    labels.extend([language] * freq)

        # Check for empty data or labels
        if not data or not labels:
            raise ValueError("Training data or labels are empty. Please check your dictionary files.")

        # Create a DataFrame to check for NaN values
        df = pd.DataFrame({'data': data, 'labels': labels})
        if df.isnull().any().any():
            print(f"NaN values found in training data:\n{df[df.isnull().any(axis=1)]}")

        # Create a pipeline with CountVectorizer and MultinomialNB
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(data, labels)
        return model

    def predict_language(self, sentence):
        return self.model.predict([sentence])[0]

    def determine_language(self, sentences):
        language_counter = Counter()
        for sentence in sentences:
            dominant_language = self.predict_language(sentence)
            language_counter[dominant_language] += 1
        return language_counter.most_common(1)[0][0] if language_counter else None

    def evaluate_model(self, test_sentences, true_labels):
        """Evaluate the model and calculate accuracy, precision, recall, and F1 score."""
        predictions = [self.predict_language(sentence) for sentence in test_sentences]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

        return accuracy, precision, recall, f1

def run_preprocessing(preprocess_script):
    """Function to run the preprocessing script."""
    try:
        # Using subprocess to run the preprocessing script
        subprocess.run(["python", preprocess_script], check=True)
        print("Preprocessing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the preprocessing: {e}")
    except FileNotFoundError:
        print("Error: preprocess.py file not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Run preprocessing first
    preprocess_script = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/preprocess.py"
    run_preprocessing(preprocess_script)

    # Proceed with dictionary generation
    preprocessed_dir = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/preprocessed"
    dictionary_dir = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary"

    generator = DictionaryGenerator(preprocessed_dir, dictionary_dir)
    languages = ['tagalog', 'bikol', 'cebuano']

    for language in languages:
        generator.generate_dictionary(language)

    # Now, initialize language identification
    language_id = LanguageIdentification(dictionary_dir)

    # Replace with actual sentences and corresponding true labels for evaluation
    test_sentences = ["tatalon sisigaw iiyak", "example sentence in another language"]
    true_labels = ["tagalog", "bikol"]  # Update this list based on your test sentences

    accuracy, precision, recall, f1 = language_id.evaluate_model(test_sentences, true_labels)

    print(f"Model Evaluation Metrics:\n"
          f"Accuracy: {accuracy:.2f}\n"
          f"Precision: {precision:.2f}\n"
          f"Recall: {recall:.2f}\n"
          f"F1 Score: {f1:.2f}")

    # Determine the dominant language from sentences
    sentences = ["tatalon sisigaw iiyak"]  # Replace with actual sentences
    dominant_language = language_id.determine_language(sentences)
    print(f"The dominant language is: {dominant_language}")
