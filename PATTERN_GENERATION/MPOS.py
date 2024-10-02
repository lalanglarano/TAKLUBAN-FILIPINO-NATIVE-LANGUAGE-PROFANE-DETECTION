import pandas as pd
import os

# Function to define POS Masking Rules
def get_pos_masking_rules():
    return {
        'NN': 'NOUN',   # Nouns
        'NNS': 'NOUN',
        'VB': 'VERB',   # Verbs
        'VBD': 'VERB',
        'VBG': 'VERB',
        'VBN': 'VERB',
        'VBP': 'VERB',
        'VBZ': 'VERB',
        'JJ': 'ADJ',    # Adjectives
        'RB': 'ADV',    # Adverbs
        'IN': 'PREP',   # Prepositions
        'DT': 'DET',    # Determiners
        'FW': 'FW',     # Foreign Words
    }

def extract_and_mask_pos_words(language):
    # Define the paths
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    results_folder = f"{base_path}/Results"
    input_file = f"{results_folder}/pos_tagged/FPOSTagged_{language}.csv"
    output_dir = f"{results_folder}/ExtractedMaskedPOS"
    output_file = f"{output_dir}/MaskedPOS_{language}.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV input file
    df = pd.read_csv(input_file)
    
    masked_sentences = []
    pos_masking = get_pos_masking_rules()  # Get the POS masking rules
    
    for sentence in df['pos_tagged']:
        words = sentence.split()
        
        masked_sentence = []
        for word in words:
            word_split = word.split('|')
            if len(word_split) == 2:  
                token, pos_tag = word_split[0], word_split[1]
                masked_tag = pos_masking.get(pos_tag, pos_tag)  
                masked_sentence.append(f"{token}|{masked_tag}")
            else:
                print(f"Warning: Skipping word without POS tag: {word}")
        
        masked_sentences.append(" ".join(masked_sentence))
    
    # Save masked sentences to CSV
    df['masked_pos'] = masked_sentences
    df.to_csv(output_file, index=False)
    
    print(f"Masked POS for {language} saved to {output_file}")

languages = ['bikol', 'cebuano', 'tagalog']

for lang in languages:
    extract_and_mask_pos_words(lang)
