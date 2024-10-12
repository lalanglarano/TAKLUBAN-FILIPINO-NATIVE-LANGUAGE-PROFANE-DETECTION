import os
import pandas as pd
from nltk.tag.stanford import StanfordPOSTagger

class POSTagger:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results/DATASETOFREGEX"
        self.input_file = f"{base_path}/Results/preprocessed/preprocessed_{language}_sentence_profane.csv"
        self.output_dir = results_folder
        self.output_file = f"{self.output_dir}/Tagged_{language}.csv"

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load the preprocessed data
        self.data = pd.read_csv(self.input_file, names=['sentence', 'label'])  # Assume 'sentence' and 'label' columns
        print(f"Loaded preprocessed data for {language}. Number of sentences: {len(self.data)}")

        # Set up the Stanford POS Tagger with increased memory allocation to 6 GB
        self.tagger = StanfordPOSTagger(
            model_filename='Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger',
            path_to_jar='Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar',
            java_options='-mx6144m'  # Set maximum Java heap size to 6 GB
        )

    def pos_tag_text(self, text):
        # Perform POS tagging
        try:
            tokens = text.split()
            # Use word|tag format instead of word/tag
            pos_tags = self.tagger.tag(tokens)
            pos_tagged_text = ' '.join([f"{word}|{tag}" for word, tag in pos_tags])  # Changed format to word|tag
            return pos_tagged_text
        except Exception as e:
            print(f"Error during POS tagging: {e}")
            return text

    def pos_tag_sentences(self, batch_size=10):
        try:
            for i in range(0, len(self.data), batch_size):
                batch = self.data.iloc[i:i+batch_size].copy()  # Copy the batch to avoid the warning

                # Apply POS tagging only to the 'sentence' column
                batch.loc[:, 'pos_tagged'] = batch['sentence'].apply(self.pos_tag_text)  # Use .loc to avoid SettingWithCopyWarning

                # Save both 'pos_tagged' and 'label' columns to the output CSV
                batch[['pos_tagged', 'label']].to_csv(self.output_file, mode='a', index=False, header=(i == 0))
                print(f"Processed batch {i//batch_size + 1} of {len(self.data) // batch_size + 1}")
            
            print(f"POS tagging complete. Results saved to {self.output_file}.")
        except Exception as e:
            print(f"An error occurred during POS tagging: {e}")

# POS tag files for multiple languages
languages = ['bikol', 'cebuano', 'tagalog']

for language in languages:
    pos_tagger = POSTagger(language)
    pos_tagger.pos_tag_sentences()
