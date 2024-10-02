import pandas as pd

# Load the CSV files
profane_data_path = 'Results/dataset/PROFANE WITH POSTAG.csv'
sentence_data_path = 'Results/pos_tagged/FPOSTagged_bikol.csv'

# Load the profane word with POS tags data
pos_data = pd.read_csv(profane_data_path, header=None)

# Load the sentence data
sentence_data = pd.read_csv(sentence_data_path, header=None, names=['sentence'])

# Function to update the FW tags based on the provided CSV file
def update_tags_final(sentence, pos_data):
    updated_sentence = []
    # Split the sentence into individual word|tag pairs
    words_tags = sentence.split(' ')
    
    for word_tag in words_tags:
        # Separate the word and the tag
        if '|' in word_tag:
            word, tag = word_tag.split('|')
            # Clean the word by stripping unnecessary characters including leading slashes
            word_cleaned = word.strip().lstrip('/').lower()
            # Check if the current tag is FW
            if tag == 'FW':
                # Look for the word in the CSV file to find its correct POS tag
                correct_tag_row = pos_data[pos_data[0].str.lower().str.strip() == word_cleaned]
                if not correct_tag_row.empty:
                    # If a match is found, use the POS Tag from the CSV file
                    correct_tag = correct_tag_row.iloc[0, 1]
                    updated_sentence.append(f"{word}|{correct_tag}")
                else:
                    # If no match is found, keep the tag as FW
                    updated_sentence.append(f"{word}|FW")
            else:
                updated_sentence.append(word_tag)
        else:
            updated_sentence.append(word_tag)
    
    return ' '.join(updated_sentence)

# Apply the function to each sentence in the dataset
sentence_data['updated_sentence'] = sentence_data['sentence'].apply(lambda x: update_tags_final(x, pos_data))

# Save the updated sentences to a new CSV file
output_path = 'C:/Users/Annalyn/TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/PFW/dataset_bikol_updated.csv'
sentence_data.to_csv(output_path, index=False)

print(f"Updated sentences saved to {output_path}")

