import nlu
import os

def lemmatize_sentence(sentence):
    """
    Calls the nlu lemmatizer to lemmatize a sentence and returns the lemmatized string.
    """
    try:
        # Check if the sentence is enclosed in single quotation marks with a comma before the closing mark
        if sentence.startswith('"') and sentence.endswith('"') and ',' in sentence:
            sentence = sentence[1:-2]  # Remove the opening and closing quotation marks and the comma

        # Load the lemmatizer model
        lemmatizer = nlu.load("tl.lemma")

        # Use the model to predict lemmatized output
        result = lemmatizer.predict(sentence)

        # Extract the lemmatized text from the result
        lemmatized_string = result['lemma'].values[0]

        # Add back the quotation marks if they were removed
        if ',' in sentence:
            lemmatized_string = '"' + lemmatized_string + '"'

        return lemmatized_string

    except Exception as e:
        print(f"Exception occurred during lemmatization: {e}")
        return sentence

def process_txt_in_batches(input_file, output_file, batch_size=10):
    """
    Reads a text file, lemmatizes sentences in batches, and writes the results to a new text file by batch.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Open the input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            sentences = infile.readlines()

        # Process the file in batches
        for i in range(0, len(sentences), batch_size):
            # Get the current batch of sentences
            batch = sentences[i:i+batch_size]

            # Lemmatize each sentence in the batch
            lemmatized_batch = [lemmatize_sentence(sentence.strip()) for sentence in batch]

            # Write the lemmatized batch to the output file
            with open(output_file, 'a', encoding='utf-8') as outfile:
                outfile.write('\n'.join(lemmatized_batch) + '\n')

            print(f"Processed {len(lemmatized_batch)} sentences, results appended to {output_file}")

    except Exception as e:
        print(f"Exception occurred during TXT batch processing: {e}")

if __name__ == "__main__":
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    results_folder = f"{base_path}/Results"
    languages = ['tagalog']  #Languages to process

    for language in languages:
        input_file = f"{results_folder}/preprocessed/preprocessed_{language}_sentence_profane.txt"
        output_dir = f"{results_folder}/lemmatized/"
        output_file = f"{output_dir}/lemmatize_{language}.txt"
        
        print(f"Processing {language} in batches...")
        process_txt_in_batches(input_file, output_file, batch_size=10)  #Batch size
