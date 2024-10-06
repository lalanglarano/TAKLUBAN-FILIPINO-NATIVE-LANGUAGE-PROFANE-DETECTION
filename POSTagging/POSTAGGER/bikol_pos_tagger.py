import os
import pandas as pd
import re

class POSTagger:
    def __init__(self):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/lemmatized/lemmatize_bikol.csv"
        self.output_dir = f"{results_folder}/pos_tagged/"
        self.output_file = f"{self.output_dir}/FPOSTagged_bikol.csv"

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load the lemmatized data
        self.data = pd.read_csv(self.input_file, names=['lemmatized'])
        print(f"Loaded lemmatized data for Bikol. Number of sentences: {len(self.data)}")

    def apply_custom_rules(self, token):
        """
        Apply custom regex rules to handle specific Bikol structures.
        """

        # Define regex patterns for different parts of speech (case-insensitive)
        patterns = {
            'VB': r'\b(ma|mag|nag|mang|pinag|pa|ka)[a-zA-Z]+\b',  # Bikol verb markers
            'NN': r'\b[a-zA-Z]+on\b|\b[a-zA-Z]+an\b|\b[a-zA-Z]+(ta|hon|lay|li)[a-zA-Z]*\b',  # Bikol nouns
            'JJ': r'\b(a|ka|mala)[a-zA-Z]+on\b|\bpinaka[a-zA-Z]+\b',  # Bikol adjectives
            'PRP': r'\bako|ikaw|siya|kami|kita|sinda|niya|ninda|niato|nato|saindo\b',  # Bikol pronouns
            'DT': r'\bang|mga|si|sa|kan|kun\b',  # Bikol determiners
            'RB': r'\b(dakul|gad|hala|dai|maya|sira|sinya|urug)\b',  # Bikol adverbs
            'CC': r'\bog|pero|kundi\b',  # Conjunctions like "og", "pero"
            'IN': r'\bpara|paagi|asin|kan\b',  # Prepositions like "para", "paagi"
            'CD': r'\bisa|duwa|tulo|apat|lima|anum|pito|walo|siyam|sampulo\b',  # Numbers in Bikol
            'EX': r'\bmay\b|\bmayda\b',  # Existentials
            'NNC': r'\bENOT|IGWA\b'  # Custom Bikol nouns
        }

        # Apply regex patterns to the token with case-insensitivity
        for tag, pattern in patterns.items():
            if re.fullmatch(pattern, token, flags=re.IGNORECASE):
                print(f"Matched: {token} -> {tag}")  # Debugging print
                return f"{token}/{tag}"

        # If no pattern matched, return token with 'UNK' tag
        print(f"No match for: {token}")  # Debugging print for unmatched tokens
        return f"{token}/UNK"

    def pos_tag_text(self, text):
        # Ensure the text is a string and not a float or other type
        if isinstance(text, str):
            # Convert text to lowercase before tokenizing
            text = text.lower()
            print(f"Processing text: {text}")  # Debugging print
            
            # Tokenize the text
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
                batch['lemmatized'] = batch['lemmatized'].fillna('').astype(str)

                # Apply POS tagging
                batch['pos_tagged'] = batch['lemmatized'].apply(self.pos_tag_text)

                # Save the output to the CSV file
                batch[['pos_tagged']].to_csv(self.output_file, mode='a', index=False, header=(i == 0))
                print(f"Processed batch {i//batch_size + 1} of {len(self.data) // batch_size + 1}")

            print(f"POS tagging complete. Results saved to {self.output_file}.")
        except Exception as e:
            print(f"An error occurred during POS tagging: {e}")

if __name__ == "__main__":
    pos_tagger = POSTagger()
    pos_tagger.pos_tag_sentences()
