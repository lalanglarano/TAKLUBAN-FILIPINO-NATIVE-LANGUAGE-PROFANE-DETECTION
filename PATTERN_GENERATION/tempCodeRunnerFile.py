import pandas as pd
from collections import Counter
from nltk.util import ngrams

# Load POS-tagged files
def load_pos_tagged_files():
    # Load the POS tagged files
    tagalog_df = pd.read_csv('Results/pos_tagged/FPOSTagged_tagalog.csv')
    bikol_df = pd.read_csv('Results/pos_tagged/FPOSTagged_bikol.csv')
    cebuano_df = pd.read_csv('Results/pos_tagged/FPOSTagged_cebuano.csv')
    
    return tagalog_df, bikol_df, cebuano_df

# Extract N-grams from POS sequences
def extract_pos_ngrams(df, n=2):
    if 'POS' not in df.columns:
        raise KeyError("The dataframe does not contain a 'POS' column")
    pos_sequences = df['POS'].values.tolist()
    pos_ngrams = [ngrams(sequence.split(), n) for sequence in pos_sequences]  # Create n-grams
    return pos_ngrams

# Analyze common n-grams
def analyze_ngrams(pos_ngrams):
    ngram_counter = Counter()
    for pos_ngram_list in pos_ngrams:
        ngram_counter.update(pos_ngram_list)
    
    # Return the most common n-grams
    return ngram_counter.most_common()

# Define rules from common patterns
def predefined_rules_from_ngrams(common_ngrams):
    predefined_rules = []
    
    for ngram, freq in common_ngrams:
        # Create rules from patterns, e.g., sequences of VERB, NOUN, etc.
        rule = f"If sequence {ngram} appears, apply predefined rule."
        predefined_rules.append((ngram, rule, freq))
    
    return predefined_rules

    for ngram, rule, _ in predefined_rules:ns
def detect_profanity(sentence_pos_tags, predefined_rules):
    for ngram, rule, freq in predefined_rules:
        if ngram in ngrams(sentence_pos_tags, len(ngram)):
            return f"Profanity rule triggered: {rule}"
    return "No profanity detected"

# Main processing function
def main():
    tagalog_df, bikol_df, cebuano_df = load_pos_tagged_files()
    
    # Extract n-grams from Bikol, Tagalog, and Cebuano POS-tagged data
    tagalog_ngrams = extract_pos_ngrams(tagalog_df, n=2)
    bikol_ngrams = extract_pos_ngrams(bikol_df, n=2)
    cebuano_ngrams = extract_pos_ngrams(cebuano_df, n=2)
    
    # Analyze the most common n-grams
    tagalog_common_ngrams = analyze_ngrams(tagalog_ngrams)
    bikol_common_ngrams = analyze_ngrams(bikol_ngrams)
    cebuano_common_ngrams = analyze_ngrams(cebuano_ngrams)
    predefined_rules_from_ngrams(tagalog_common_ngrams)
    # Create predefined rules based on the most common n-grams
    predefined_rules_from_ngrams(cebuano_common_ngrams)
    bikol_rules = predefined_rules_from_ngrams(bikol_common_ngrams)
    cebuano_rules = predefined_rules_from_ngrams(cebuano_common_ngrams)
    
    # Example: Use rules to detect profanity in a new sentence
    sentence_pos_tags = ["NOM", "VERB", "NOUN", "ADJ"]  # Example POS tags for a sentence
    profanity_result = detect_profanity(sentence_pos_tags, bikol_rules)
    
    print(profanity_result)

if __name__ == "__main__":
    main()
