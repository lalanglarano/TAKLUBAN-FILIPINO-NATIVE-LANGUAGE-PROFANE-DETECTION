import os
import joblib
from SPOStagging import POSTagger as SPOSTagger
from cebuano_pos_tagger import POSTagger as CebuanoTagger
from bikol_pos_tagger import POSTagger as BikolTagger

def save_model(model, file_path):
    """Save the POS process or any object to a pickle file."""
    with open(file_path, 'wb') as file:
        joblib.dump(model, file)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """Load a model or object from a pickle file."""
    with open(file_path, 'rb') as file:
        return joblib.load(file)

def step1_spost_tagging(language):
    """Step 1: Run SPOSTAGGING for a specific language."""
    print(f"Running Stanford POS Tagger for {language}...")
    pos_tagger = SPOSTagger(language)
    pos_tagger.pos_tag_sentences()
    return pos_tagger

def step2_cebuano_regex():
    """Step 2: Apply regex rules for Cebuano POS refinement."""
    print("Running regex-based POS refinement for Cebuano...")
    pos_tagger = CebuanoTagger()
    pos_tagger.pos_tag_sentences()
    return pos_tagger

def step2_bikol_regex():
    """Step 2: Apply regex rules for Bikol POS refinement."""
    print("Running regex-based POS refinement for Bikol...")
    pos_tagger = BikolTagger()
    pos_tagger.pos_tag_sentences()
    return pos_tagger

def main():
    # Step 1: POS tagging for Tagalog, Cebuano, and Bikol
    languages = ['bikol', 'cebuano', 'tagalog']
    for language in languages:
        pos_model = step1_spost_tagging(language)
        # Save the POS tagging model for each language
        save_model(pos_model, f"../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/models/pos_{language}_model.pkl")

    # Step 2: Apply regex refinements and save the final models
    cebuano_model = step2_cebuano_regex()
    save_model(cebuano_model, "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/models/tagged_cebuano_model.pkl")

    bikol_model = step2_bikol_regex()
    save_model(bikol_model, "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/models/tagged_bikol_model.pkl")

    print("All steps completed and models saved.")

if __name__ == "__main__":
    main()
