import os
import csv
import joblib
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Configuration Dictionary
config = {
    "base_path": "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION",
    "preprocessed_dir": "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/PFW",
    "dictionary_dir": "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary",
    "english_dict_path": "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary/english_dictionary.csv",
    "model_path": "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"
}

class DictionaryGenerator:
    """Preprocessing and Dictionary Management."""

    def __init__(self, config):
        self.config = config
        self.noise_words = self.initialize_noise_words()

    def initialize_noise_words(self):
        """Load predefined noise words and remove common overlaps."""
        noise_words = {
            'Tagalog': {"na", "nang", "ng", "mga", "kung", "yan", "ko", "po", "ka", "ikaw"},
            'Bikol': {"ngani", "ini", "kang", "iyo", "hali", "baga", "ho", "mo"},
            'Cebuano': {"dayon", "gani", "kana", "mao", "pud", "bitaw"}
        }
        noise_words['English'] = self.load_english_noise_words()
        self.clean_common_words(noise_words)
        return noise_words

    def load_english_noise_words(self):
        """Load noise words from English dictionary."""
        noise_words = set()
        try:
            with open(self.config['english_dict_path'], 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    noise_words.add(row[0].strip().lower())
        except FileNotFoundError:
            print(f"Error: English dictionary file not found at {self.config['english_dict_path']}.")
        return noise_words

    def clean_common_words(self, noise_words):
        """Remove overlapping words among noise word sets."""
        common_words = (noise_words['Tagalog'] | noise_words['Bikol'] | noise_words['Cebuano']) & noise_words['English']
        noise_words['English'] -= common_words

    def generate_dictionaries(self, languages):
        """Generate frequency dictionaries for each language."""
        os.makedirs(self.config['dictionary_dir'], exist_ok=True)
        for language in languages:
            file_path = os.path.join(self.config['preprocessed_dir'], f"preprocessed_{language}.csv")
            if not os.path.exists(file_path):
                print(f"Error: Preprocessed file for {language} not found at {file_path}.")
                continue
            word_counts = self.create_word_count(file_path, language)
            self.save_dictionary(word_counts, language)

    def create_word_count(self, file_path, language):
        """Count word frequencies, excluding noise words."""
        word_count = Counter()
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    words = row[0].split()
                    filtered_words = [w.lower() for w in words if w.lower() not in self.noise_words[language.capitalize()]]
                    word_count.update(filtered_words)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        return word_count

    def save_dictionary(self, word_count, language):
        """Save word counts to a CSV file."""
        output_file = os.path.join(self.config['dictionary_dir'], f"{language}_dictionary.csv")
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["word", "frequency"])
            for word, count in sorted(word_count.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([word, count])
        print(f"Dictionary for {language} saved to {output_file}")

class LanguageIdentification:
    """Model training and evaluation."""

    def __init__(self, config):
        self.config = config
        self.model = None

    def train_model(self):
        """Train the model and perform a grid search."""
        print("Preparing data...")
        data, labels = self.prepare_data()
        if not data or not labels:
            raise ValueError("No data available for training. Please check your dictionaries.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        
        print("Training data prepared.")
        
        # Define pipeline and parameter distributions
        pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 3)), MultinomialNB())
        param_distributions = {
            'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'multinomialnb__alpha': [0.1, 0.5, 1.0]
        }

        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            cv=3,
            n_iter=10,
            scoring='accuracy',
            error_score='raise',
            random_state=42
        )
        
        print("Starting model training...")
        search.fit(X_train, y_train)
        print("Training completed.")
        
        # Save the best model
        self.model = search.best_estimator_
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print("Model evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        return X_test, y_test

    def prepare_data(self):
        """Prepare data from dictionaries for training."""
        data, labels = [], []
        for language in ['tagalog', 'bikol', 'cebuano']:
            file_path = os.path.join(self.config['dictionary_dir'], f"{language}_dictionary.csv")
            if not os.path.exists(file_path):
                print(f"Dictionary file for {language} not found.")
                continue
            
            try:
                df = pd.read_csv(file_path)
                df = df.dropna(subset=['word', 'frequency'])
                df['word'] = df['word'].astype(str)
                
                for _, row in df.iterrows():
                    data.extend([row['word']] * int(row['frequency']))
                    labels.extend([language] * int(row['frequency']))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return data, labels

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
        self.plot_confusion_matrix(y_test, y_pred)

    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred, labels=['tagalog', 'bikol', 'cebuano'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tagalog', 'Bikol', 'Cebuano'], yticklabels=['Tagalog', 'Bikol', 'Cebuano'])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    def predict_sentence(self):
        """Ask the user for a sentence and predict its language."""
        if not self.model:
            print("Error: The model is not trained. Please train the model first.")
            return
        
        while True:
            user_input = input("Enter a sentence to predict its language: ").strip()
            if not user_input:
                print("No input provided. Exiting prediction.")
                break
            
            # Make a prediction
            prediction = self.model.predict([user_input])
            print(f"The predicted language is: {prediction[0].capitalize()}")
            
            repeat = input("Do you want to predict another sentence? (yes/no): ").strip().lower()
            if repeat != 'yes':
                print("Exiting the prediction tool.")
                break

# Main Execution
if __name__ == "__main__":
    print("Initializing preprocessing pipeline...")
    pipeline = PreprocessingPipeline(config)
    pipeline.generate_dictionaries(['Tagalog', 'Bikol', 'Cebuano'])
    
    print("Training language model...")
    model = LanguageModel(config)
    X_test, y_test = model.train_model()
    
    print("Predicting language...")
    model.predict_sentence()
    
    # print("Evaluating model...")
    # model.evaluate_model(X_test, y_test)
    

