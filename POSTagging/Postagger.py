import os
import pandas as pd
import re
from nltk.tag.stanford import StanfordPOSTagger

class StanfordPOSTaggerWrapper:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        self.input_file = f"{base_path}/Results/preprocessed/preprocessed_{language}_sentence_profane.csv"
        self.output_file = f"{base_path}/Results/DATASETOFREGEX/Tagged_{language}.csv"

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

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
        try:
            tokens = text.split()
            pos_tags = self.tagger.tag(tokens)
            return ' '.join([f"{word}|{tag}" for word, tag in pos_tags])  # Use word|tag format
        except Exception as e:
            print(f"Error during POS tagging: {e}")
            return text

    def pos_tag_sentences(self, batch_size=10):
        try:
            for i in range(0, len(self.data), batch_size):
                batch = self.data.iloc[i:i + batch_size].copy()
                batch['pos_tagged'] = batch['sentence'].apply(self.pos_tag_text)

                # Save tagged sentences
                batch[['pos_tagged', 'label']].to_csv(self.output_file, mode='a', index=False, header=(i == 0))
                print(f"Processed batch {i//batch_size + 1} of {len(self.data) // batch_size + 1}")

            print(f"POS tagging complete. Results saved to {self.output_file}.")
        except Exception as e:
            print(f"An error occurred during POS tagging: {e}")


class POSTagger:
    def __init__(self, language):
        self.language = language.lower()
        self.output_file = 'POSTagging/POSTAGGER/POSData.csv'
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Initialize StanfordPOSTaggerWrapper for supported languages
        if self.language in ['tagalog', 'cebuano', 'bikol']:
            self.stanford_tagger = StanfordPOSTaggerWrapper(language)

    @staticmethod
    def clean_token(token):
        token = token.lstrip('/')  # Remove leading '/' if present
        if '|' not in token:
            print(f"Invalid token format, no '|': {token}")
            return token
        word, tag = map(str.strip, token.split('|', 1))  # Split only once at '|'
        return f"{word}|{tag}"

    def language_rules(self, token):
        token = self.clean_token(token)
        word, current_tag = token.split('|', 1)

        patterns = {
            'cebuano': {
                'VB': r'\b(mag|nag|mi|mo|mu|mang|manag|man)[a-zA-Z]+\b',
                'NNC': r'\b([a-zA-Z]+on|[a-zA-Z]+an)\b',
                'NNCA': r'\b(ka|pang)[a-zA-Z]+an\b',
                'NNPL': r'\bmga\s+[a-zA-Z]+\b',
                'JJD': r'\b(ma|ka)[a-zA-Z]+an\b',
                'JJCM': r'\bmas\s+[a-zA-Z]+\b',
                'PRP': r'\bako|ikaw|siya|kami|kita|sila\b',
                'DT': r'\bang|bang|mga\b',
                'CCP': r'\bug|o|kundi\b',
            },
            'bikol': {
                'VB': r'\b(MA|MAG|NAG|MANG|PINAG|PA|KA)[a-zA-Z]+\b',
                'NN': r'\b[a-zA-Z]+on\b|\b[a-zA-Z]+an\b|\b[a-zA-Z]+(TA|HON|LAY|LI)[a-zA-Z]*\b',
                'JJ': r'\b(A|KA|MALA)[a-zA-Z]+on\b|\bPINAKA[a-zA-Z]+\b',
                'PRP': r'\bAKO|IKAW|SIYA|KAMI|KITA|SINDA|NIYA|NINDA|NIATO|NATO|SAINDO\b',
                'DT': r'\bANG|MGA|SI|SA|KAN|KUN\b',
                'RB': r'\b(DAKUL|GAD|HALA|DAI|MAYA|SIRA|SINYA|URUG)\b',
                'CC': r'\bOG|PERO|KUNDI\b',
                'IN': r'\bPARA|PAAGI|ASIN|KAN\b',
                'CD': r'\bSARO|DUWA|TULO|APAT|LIMA|ANOM|PITO|WALO|SIYAM|SAMPULO\b',
            }
        }

        language_patterns = patterns.get(self.language, {})
        for tag, pattern in language_patterns.items():
            if re.fullmatch(pattern, word, flags=re.IGNORECASE):
                print(f"Matched: {word} -> {tag}")
                return f"{word}|{tag}"

        return token

    def pos_tag_text(self, text):
        stanford_tagged_text = self.stanford_tagger.pos_tag_text(text)
        print(f"Stanford tagged text: {stanford_tagged_text}")

        if self.language in ['cebuano', 'bikol']:
            tokens = stanford_tagged_text.split()
            return ' '.join([self.language_rules(token) for token in tokens])

        return stanford_tagged_text

    def pos_tag_input(self, user_text):
        pos_tagged_text = self.pos_tag_text(user_text)

        # Save the tagged text to the CSV file
        with open(self.output_file, 'a') as f:
            f.write(f"{user_text},{pos_tagged_text}\n")

        print(f"POS tagged text saved to {self.output_file}")


if __name__ == "__main__":
    language = input("Enter the language to process (cebuano/bikol/tagalog): ").lower()

    if language not in ['cebuano', 'bikol', 'tagalog']:
        print(f"Unsupported language: {language}")
    else:
        pos_tagger = POSTagger(language)
        user_text = input("Enter the text you want to POS tag: ")
        pos_tagger.pos_tag_input(user_text)