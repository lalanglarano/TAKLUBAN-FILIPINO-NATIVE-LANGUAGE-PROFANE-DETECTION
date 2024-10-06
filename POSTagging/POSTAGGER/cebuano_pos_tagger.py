import os
import pandas as pd
from nltk.tag.stanford import StanfordPOSTagger
import re

class POSTagger:
    def __init__(self):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/lemmatized/lemmatize_cebuano.csv"
        self.output_dir = f"{results_folder}/pos_tagged/"
        self.output_file = f"{self.output_dir}/FPOSTagged_cebuano.csv"

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load the lemmatized data
        self.data = pd.read_csv(self.input_file, names=['lemmatized'])
        print(f"Loaded lemmatized data for Cebuano. Number of sentences: {len(self.data)}")

        # Set up the Stanford POS Tagger
        self.tagger = StanfordPOSTagger(
            model_filename='Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger',
            path_to_jar='Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
        )

    def apply_custom_rules(self, pos_tagged_text):
        """
        Apply custom regex rules to handle specific Cebuano structures.
        """
        # Define regex patterns for different parts of speech
        patterns = {
            'VB': r'\b(mag|nag|mi|ni)[a-zA-Z]+\b',  # Cebuano verb markers
            'NN': r'\b[a-zA-Z]+on\b',  # Cebuano nouns ending in "on"
            'JJ': r'\b(ma|ka)[a-zA-Z]+an\b',  # Cebuano adjectives
            'PRP': r'\bako|ikaw|siya|kami|kita|sila\b',  # Cebuano pronouns
            'DT': r'\bang|ang mga\b',  # Cebuano determiners
        }

        # Apply regex patterns to the POS tagged text
        for tag, pattern in patterns.items():
            pos_tagged_text = re.sub(pattern, lambda m: f"{m.group(0)}/{tag}", pos_tagged_text)
        
        return pos_tagged_text

    def pos_tag_text(self, text):
        # Perform POS tagging using SPOST
        try:
            tokens = text.split()
            pos_tags = self.tagger.tag(tokens)
            pos_tagged_text = ' '.join([f"{word}/{tag}" for word, tag in pos_tags])

            # Apply custom regex rules
            pos_tagged_text = self.apply_custom_rules(pos_tagged_text)

            print(f"POS-tagged text (with custom rules): {pos_tagged_text}")
            return pos_tagged_text
        except Exception as e:
            print(f"Error during POS tagging: {e}")
            return text

    def pos_tag_sentences(self, batch_size=10):
        try:
            for i in range(0, len(self.data), batch_size):
                batch = self.data.iloc[i:i+batch_size]
                batch['pos_tagged'] = batch['lemmatized'].apply(self.pos_tag_text)
                batch[['pos_tagged']].to_csv(self.output_file, mode='a', index=False, header=(i == 0))
                print(f"Processed batch {i//batch_size + 1} of {len(self.data) // batch_size + 1}")
            print(f"POS tagging complete. Results saved to {self.output_file}.")
        except Exception as e:
            print(f"An error occurred during POS tagging: {e}")


if __name__ == "__main__":
    pos_tagger = POSTagger()
    pos_tagger.pos_tag_sentences()
