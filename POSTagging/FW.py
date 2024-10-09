import pandas as pd
import os

def extract_fw_words():
    language = 'bikol'  
    # Define the paths
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    results_folder = f"{base_path}/Results"
    input_file = f"{results_folder}/PFW/dataset_{language}_updated.csv"
    output_dir = f"{results_folder}/ExtractedFW"
    output_file = f"{output_dir}/FW_{language}.csv"
    
    os.makedirs(output_dir, exist_ok=True)  
    
    try:
        # Read the CSV input file
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {input_file} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: The file {input_file} could not be parsed.")
        return

    # Print the columns for debugging
    print("Columns in the DataFrame:", df.columns.tolist())

    # Update this line with the correct column name
    column_name = 'updated_sentence'  # Replace with the correct column name

    # Check if the specified column exists
    if column_name not in df.columns:
        print(f"Error: The column '{column_name}' does not exist in the DataFrame.")
        return

    fw_words = set()  # Use a set to avoid duplicates
    
    for sentence in df[column_name]:
        words = sentence.split()
        fw_words_in_sentence = [word.split('|')[0] for word in words if '|FW' in word]
        fw_words.update(fw_words_in_sentence)  # Add found words to the set
    
    # Create a DataFrame and save it
    fw_df = pd.DataFrame(list(fw_words), columns=['FW_Word'])
    fw_df.to_csv(output_file, index=False)
    
    print(f"FW words extracted and saved to {output_file}")

extract_fw_words()
