import os
import pandas as pd
from nltk.tag.stanford import StanfordPOSTagger
import subprocess

class POSTagger:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/lemmatized/lemmatize_{language}.csv"
        self.output_dir = f"{results_folder}/pos_tagged/"
        self.output_file = f"{self.output_dir}/FPOSTagged_{language}.csv"

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load the lemmatized data
        self.data = pd.read_csv(self.input_file, names=['lemmatized'])
        print(f"Loaded lemmatized data for {language}. Number of sentences: {len(self.data)}")

        # Set up the Stanford POS Tagger
        self.tagger = StanfordPOSTagger(
            model_filename='Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger',
            path_to_jar='Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
        )

    def pos_tag_text(self, text):
        # Perform POS tagging
        try:
            tokens = text.split()
            pos_tags = self.tagger.tag(tokens)
            pos_tagged_text = ' '.join([f"{word}/{tag}" for word, tag in pos_tags])
            print(f"POS-tagged text: {pos_tagged_text}")
            return pos_tagged_text
        except Exception as e:
            print(f"Error during POS tagging: {e}")
            return text

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


def run_cebuano_and_bikol_taggers():
    """
    Function to run the external Cebuano and Bikol POS taggers.
    """
    try:
        # Run Cebuano POS Tagger
        cebuano_process = subprocess.Popen(["python", "cebuano_pos_tagger.py"])
        # Run Bikol POS Tagger
        bikol_process = subprocess.Popen(["python", "bikol_pos_tagger.py"])

        # Wait for both processes to finish
        cebuano_process.wait()
        bikol_process.wait()

        print("Cebuano and Bikol POS tagging processes completed.")
    except Exception as e:
        print(f"Error running Cebuano and Bikol taggers: {e}")


if __name__ == "__main__":
    # Perform Tagalog POS tagging in the main script
    pos_tagger = POSTagger('tagalog')
    pos_tagger.pos_tag_sentences()

    # Run external Cebuano and Bikol taggers concurrently
    run_cebuano_and_bikol_taggers()
