import pandas as pd
import os

def extract_fw_words():
    language = 'tagalog'  
    # Define the paths
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    results_folder = f"{base_path}/Results"
    input_file = f"{results_folder}/pos_tagged/FPOSTagged_{language}.csv"
    output_dir = f"{results_folder}/ExtractedFW"
    output_file = f"{output_dir}/FW_{language}.csv"
    
    os.makedirs(output_dir, exist_ok=True)  
    
    # Read the CSV input file
    df = pd.read_csv(input_file)
    
    fw_words = []
    
    for sentence in df['pos_tagged']:
        words = sentence.split()
        fw_words_in_sentence = [word.split('|')[0] for word in words if '|FW' in word]
        fw_words.extend(fw_words_in_sentence)
    
    fw_df = pd.DataFrame(fw_words, columns=['FW_Word'])
    fw_df.to_csv(output_file, index=False)
    
    print(f"FW words extracted and saved to {output_file}")

extract_fw_words()
