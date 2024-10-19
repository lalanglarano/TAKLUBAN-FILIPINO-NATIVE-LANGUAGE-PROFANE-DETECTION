import os
import re

class POSTagger:
    def __init__(self):
        # Define patterns for parts of speech (regex for Cebuano)
        self.patterns = {
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

    def tag_user_input(self):
        """
        Get input text from the user and perform POS tagging.
        """
        user_text = input("Enter a sentence or paragraph for POS tagging: ")
        tagged_text = self.pos_tag_text(user_text)
        print(f"POS-tagged text: {tagged_text}")
        self.save_to_file(tagged_text)

    def save_to_file(self, tagged_text):
        """
        Save the POS-tagged result to a CSV file.
        """
        output_file = "POSData.csv"
        with open(output_file, 'a') as f:
            f.write(tagged_text + '\n')
        print(f"POS-tagged text saved to {output_file}.")

if __name__ == "__main__":
    pos_tagger = POSTagger()
    pos_tagger.tag_user_input()
