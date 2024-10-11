import os
import pandas as pd
import re

class POSTagger:
    def __init__(self):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/pos_tagged/FPOSTagged_cebuano.csv"
        self.output_dir = f"{results_folder}/DATASETOFREGEX"
        self.output_file = f"{self.output_dir}/Tagged_cebuano.csv"

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Check if the input file exists
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        # Load the data
        self.data = pd.read_csv(self.input_file)
        print(f"Loaded data for Bikol. Number of sentences: {len(self.data)}")

        # Check if the 'pos_tagged' column exists
        if 'pos_tagged' not in self.data.columns:
            raise ValueError(f"'pos_tagged' column not found in the input file. Available columns: {self.data.columns}")

    def clean_token(self, token):
        """
        Clean the token by removing unwanted characters like slashes and trimming extra spaces.
        Expected format: word|tag
        """

        # Remove any leading/trailing spaces or slashes
        token = token.strip()

        # Remove leading '/' if present before the word
        if token.startswith('/'):
            token = token[1:]

        # Ensure proper formatting of 'word|tag'
        if '|' not in token:
            print(f"Invalid token format, no '|': {token}")
            return token  # Return as-is if formatting is wrong

        word, tag = token.split('|', 1)  # Split only once at '|'
        word = word.strip()  # Clean up spaces around the word
        tag = tag.strip()  # Clean up spaces around the tag

        # Return the cleaned token
        return f"{word}|{tag}"

    def apply_custom_rules(self, token):
        """
        Apply custom regex rules to handle specific Bikol structures and update the POS tag if a match is found.
        """

        # First, clean the token
        token = self.clean_token(token)

        # Define regex patterns for different parts of speech (case-insensitive)
        patterns = {
            'VB': r'\b(mag|nag|mi|mo|mu|mang|manag|man)[a-zA-Z]+\b',  # Cebuano verb markers (actor/object focus)
            'NNC': r'\b([a-zA-Z]+on|[a-zA-Z]+an)\b',  # Cebuano nouns ending in "on" or "an"
            'NNCA': r'\b(ka|pang)[a-zA-Z]+an\b',  # Cebuano nouns with affixes
            'NNPL': r'\bmga\s+[a-zA-Z]+\b',  # Plural nouns prefixed by "mga"
            'JJD': r'\b(ma|ka)[a-zA-Z]+an\b',  # Cebuano adjectives with "ma" or "ka"
            'JJCM': r'\bmas\s+[a-zA-Z]+\b',  # Cebuano comparative adjectives
            'PRP': r'\bako|ikaw|siya|kami|kita|sila\b',  # Cebuano pronouns
            'RBF': r'\bkanunay|pirme|usahay|permi\b',  # Frequency adverbs
            'RBI': r'\b(kaayo|pagka)[a-zA-Z]+\b',  # Intensifying adverbs with "kaayo" or "pagka"
            'RBR': r'\b([a-zA-Z]+)-\1\b',  # Repeated/reduplicated words (emphasis)
            'DT': r'\bang|bang|mga\b',  # Cebuano determiners
            'CCP': r'\bug|o|kundi\b',  # Coordinating conjunctions
            'CCL': r'\bkay|tungod|bisan\b',  # Subordinating conjunctions
        }

        # Split the cleaned token into word and tag
        try:
            word, current_tag = token.split('|')
        except ValueError:
            print(f"Invalid token format after cleaning: {token}")
            return token  # Return as-is if formatting is still wrong

        # Apply regex patterns to the word with case-insensitivity
        for tag, pattern in patterns.items():
            if re.fullmatch(pattern, word, flags=re.IGNORECASE):
                print(f"Matched: {word} -> {tag}")  # Debugging print
                return f"{word}|{tag}"  # Update the tag

        # If no pattern matched, return the original token with its current tag
        print(f"No match for: {word}")  # Debugging print for unmatched tokens
        return token

    def pos_tag_text(self, text):
        # Ensure the text is a string and not a float or other type
        if isinstance(text, str):
            print(f"Processing text: {text}")  # Debugging print
            
            # Tokenize the text by splitting on spaces
            tokens = text.split()
            print(f"Tokens: {tokens}")  # Debugging print to check tokens

            # Apply custom rules to each token
            pos_tagged_tokens = [self.apply_custom_rules(token) for token in tokens]

            pos_tagged_text = ' '.join(pos_tagged_tokens)
            print(f"POS-tagged text: {pos_tagged_text}")  # Debugging print
            return pos_tagged_text
        else:
            # Handle non-string values (e.g., NaN) by returning an appropriate tag
            print(f"Invalid text: {text} (not a string)")  # Debugging print for invalid values
            return 'UNK'

    def pos_tag_sentences(self, batch_size=10):
        try:
            for i in range(0, len(self.data), batch_size):
                # Create a copy of the batch to avoid the warning
                batch = self.data.iloc[i:i+batch_size].copy()

                # Fill missing values and convert to string
                batch['pos_tagged'] = batch['pos_tagged'].fillna('').astype(str)

                # Apply POS tagging
                batch['pos_tagged'] = batch['pos_tagged'].apply(self.pos_tag_text)

                # Save the output to the CSV file
                batch[['pos_tagged']].to_csv(self.output_file, mode='a', index=False, header=(i == 0))
                print(f"Processed batch {i//batch_size + 1} of {len(self.data) // batch_size + 1}")

            print(f"POS tagging complete. Results saved to {self.output_file}.")
        except Exception as e:
            print(f"An error occurred during POS tagging: {e}")

if __name__ == "__main__":
    pos_tagger = POSTagger()
    pos_tagger.pos_tag_sentences()
