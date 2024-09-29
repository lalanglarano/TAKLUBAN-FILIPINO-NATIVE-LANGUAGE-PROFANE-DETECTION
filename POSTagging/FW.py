import pandas as pd
import os

def extract_fw_words():
    language = 'bikol'  # Use a string instead of a list for a single language
    # Define the paths
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    results_folder = f"{base_path}/Results"
    input_file = f"{results_folder}/pos_tagged/FPOSTagged_{language}.csv"  # Fixed path
    output_dir = f"{results_folder}/ExtractedFW"
    output_file = f"{output_dir}/FW_{language}.csv"
    
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    
    # Read the CSV input file
    df = pd.read_csv(input_file)
    
    fw_words = []
    
    for sentence in df['pos_tagged']:  # Adjusted to reference the correct column
        words = sentence.split()  # Split the sentence into words
        # Extract words tagged as FW
        fw_words_in_sentence = [word.split('|')[0] for word in words if '|FW' in word]
        fw_words.extend(fw_words_in_sentence)
    
    # Save the FW words to a new CSV file
    fw_df = pd.DataFrame(fw_words, columns=['FW_Word'])
    fw_df.to_csv(output_file, index=False)
    
    print(f"FW words extracted and saved to {output_file}")

# Example usage:
extract_fw_words()
