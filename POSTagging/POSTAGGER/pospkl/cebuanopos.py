import os
import pandas as pd
import pickle
import re

class POSTagger:
    def __init__(self, tagged_data=None):
        self.tagged_data = tagged_data  # This will store the processed POS tagged data

        # Define patterns for parts of speech (regex for Cebuano)
        self.patterns = {
            'VB': r'\b(mag|nag|mi|mo|mu|mang|manag|man)[a-zA-Z]+\b',
            'NNC': r'\b([a-zA-Z]+on|[a-zA-Z]+an)\b',
            'NNCA': r'\b(ka|pang)[a-zA-Z]+an\b',
            'NNPL': r'\bmga\s+[a-zA-Z]+\b',
            'JJD': r'\b(ma|ka)[a-zA-Z]+an\b',
            'JJCM': r'\bmas\s+[a-zA-Z]+\b',
            'PRP': r'\bako|ikaw|siya|kami|kita|sila\b',
            'RBF': r'\bkanunay|pirme|usahay|permi\b',
            'RBI': r'\b(kaayo|pagka)[a-zA-Z]+\b',
            'RBR': r'\b([a-zA-Z]+)-\1\b',
            'DT': r'\bang|bang|mga\b',
            'CCP': r'\bug|o|kundi\b',
            'CCL': r'\bkay|tungod|bisan\b',
        }

    def clean_token(self, token):
        """
        Clean the token by removing unwanted characters like slashes and trimming extra spaces.
        Expected format: word|tag
        """
        token = token.strip()
        if token.startswith('/'):
            token = token[1:]
        if '|' not in token:
            return token
        word, tag = token.split('|', 1)
        word = word.strip()
        tag = tag.strip()
        return f"{word}|{tag}"

    def apply_custom_rules(self, token):
        """
        Apply custom regex rules to handle specific Cebuano structures and update the POS tag if a match is found.
        """
        token = self.clean_token(token)
        try:
            word, current_tag = token.split('|')
        except ValueError:
            return token  # Return as-is if formatting is still wrong

        for tag, pattern in self.patterns.items():
            if re.fullmatch(pattern, word, flags=re.IGNORECASE):
                return f"{word}|{tag}"

        return token

    def pos_tag_text(self, text):
        """
        Tag the provided text by applying custom rules.
        """
        if isinstance(text, str):
            tokens = text.split()
            pos_tagged_tokens = [self.apply_custom_rules(token) for token in tokens]
            return ' '.join(pos_tagged_tokens)
        else:
            return 'UNK'

    def save_model(self, model_filename="cebuano_pos_model.pkl"):
        """
        Save the model (including tagged data) to a .pkl file.
        """
        model = {
            'tagged_data': self.tagged_data,
            'patterns': self.patterns  # You can also store the regex patterns if needed
        }
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_filename}.")

# Function to load the processed Cebuano POS-tagged data
def load_cebuano_pos_data(file_path="Results/DATASETOFREGEX/Tagged_cebuano.csv"):
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded Cebuano POS-tagged data. Number of sentences: {len(data)}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

if __name__ == "__main__":
    # Load the processed Cebuano POS data
    cebuano_data = load_cebuano_pos_data()

    if cebuano_data is not None:
        # Initialize the POS tagger with the loaded data
        pos_tagger = POSTagger(tagged_data=cebuano_data)

        # Save the model with the tagged data
        pos_tagger.save_model("cebuano_pos_model.pkl")
