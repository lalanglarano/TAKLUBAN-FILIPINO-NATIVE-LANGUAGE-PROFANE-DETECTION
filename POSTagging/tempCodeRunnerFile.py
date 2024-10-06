import os
import pandas as pd
from nltk.tag.stanford import StanfordPOSTagger

class POSTagger:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/lemmatized/lemmatize_{language}.csv"
        self.output_dir = f"{base_path}/Results/ExtractedFW/"
        self.output_file = f"{self.output_dir}/cebupos.csv"

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load the lemmatized data
        self.data = pd.read_csv(self.input_file, names=['lemmatized'])
        print(f"Loaded lemmatized data for {language}. Number of sentences: {len(self.data)}")

        # Set up the Stanford POS Tagger
        self.tagger = StanfordPOSTagger(
            model_filename=os.path.join(base_path, 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'),
            path_to_jar=os.path.join(base_path, 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar')
        )

    def pos_tag_text(self, text):
        # Perform POS tagging
        try:
            tokens = text.split()
            pos_tags = self.tagger.tag(tokens)
            pos_tagged_text = self.apply_predefined_rules(pos_tags)
            print(f"POS-tagged text with predefined rules: {pos_tagged_text}")
            return pos_tagged_text
        except Exception as e:
            print(f"Error during POS tagging: {e}")
            return text

    def apply_predefined_rules(self, pos_tags):
        # Apply predefined rules for Cebuano grammar
        new_tags = []
        for word, tag in pos_tags:
            # Rule 1: Determiners and Nouns
            if word in ['ang', 'mga', 'usa', 'sa', 'si', 'kang']:
                new_tags.append((word, 'DTC'))
            elif word in ['akong', 'imong', 'iyang']:
                new_tags.append((word, 'DTP'))
            elif word[0].isupper():
                new_tags.append((word, 'NNP'))  # Proper Noun
            elif word.startswith('nag') or word.startswith('mi') or word.startswith('mag') or word.startswith('mo'):
                new_tags.append((word, 'VBAF'))  # Active Verb
            elif word.startswith('gi') or word.startswith('ipa') or word.startswith('i') or word.startswith('ma'):
                new_tags.append((word, 'VBOF'))  # Passive Verb (for direct objects)
            elif word.endswith('an') and any(p in word for p in ['luto', 'sulat']):
                new_tags.append((word, 'VBCT'))  # Circumstantial Trigger
            elif word.startswith('i') or word.endswith('an'):
                new_tags.append((word, 'VBIT'))  # Instrumental Trigger
            elif word.startswith('mas'):
                new_tags.append((word, 'JJCM'))  # Comparative Adjective
            elif word in ['kaayo', 'gajud']:
                new_tags.append((word, 'RBI'))  # Intensifier Adverb
            elif word in ['dinhi', 'adto']:
                new_tags.append((word, 'RBR'))  # Relative Adverb
            elif word in ['ug', 'o']:
                new_tags.append((word, 'CCP'))  # Coordinating Conjunction
            elif word in ['nga', 'tungod']:
                new_tags.append((word, 'CCL'))  # Subordinating Conjunction
            elif word in ['kanunay', 'permi']:
                new_tags.append((word, 'RBF'))  # Frequency Adverb
            elif word in ['si', 'ni']:
                new_tags.append((word, 'PRS'))  # Personal Pronoun
            else:
                # Default back to Stanford POS tagger's result if no rule applies
                new_tags.append((word, tag))
        
        # Return the updated tagged text
        return ' '.join([f"{word}/{tag}" for word, tag in new_tags])

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

# Only use Cebuano dataset
language = 'cebuano'
pos_tagger = POSTagger(language)
pos_tagger.pos_tag_sentences()
