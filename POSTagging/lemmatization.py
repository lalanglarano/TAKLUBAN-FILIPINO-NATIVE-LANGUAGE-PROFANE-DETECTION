import os
import pandas as pd
import nlu

class Lemmatizer:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/preprocessed/preprocessed_{language}.csv"
        self.output_dir = f"{results_folder}/lemmatized/"
        self.output_file = f"{self.output_dir}/lemmatize_{language}.csv"

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load the preprocessed data
        self.data = pd.read_csv(self.input_file, names=['sentence'])
        print(f"Loaded data for {language}. Number of sentences: {len(self.data)}")

    def lemmatize_text(self, text):
        # Initialize NLU lemmatization pipeline
        try:
            lemmatizer = nlu.load('lemma')
            print("Lemmatizer model loaded successfully.")
        except Exception as e:
            print(f"Error loading lemmatizer model: {e}")
            return text

        # Perform lemmatization
        try:
            lemmatized_df = lemmatizer.predict(text, output_level='document')
            # Extract lemmatized text
            lemmatized_text = lemmatized_df['document_lemma'].values[0]
            print(f"Lemmatized text: {lemmatized_text}")
            return lemmatized_text
        except Exception as e:
            print(f"Error during lemmatization: {e}")
            return text

    def lemmatize_sentences(self):
        try:
            # Apply lemmatization to each sentence
            self.data['lemmatized'] = self.data['sentence'].apply(self.lemmatize_text)
            # Save the lemmatized sentences to a new CSV file
            self.data[['lemmatized']].to_csv(self.output_file, index=False, header=False)
            print(f"Lemmatization complete. Results saved to {self.output_file}.")
        except Exception as e:
            print(f"An error occurred during lemmatization: {e}")

# Lemmatize files for Tagalog, Bikol, and Cebuano
languages = ['tagalog', 'bikol', 'cebuano']

for language in languages:
    lemmatizer = Lemmatizer(language)
    lemmatizer.lemmatize_sentences()
