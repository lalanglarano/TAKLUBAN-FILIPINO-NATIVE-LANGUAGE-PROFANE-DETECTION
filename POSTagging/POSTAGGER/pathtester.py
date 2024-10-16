import os
import joblib
import pandas as pd

def load_model(file_path):
    """Load a model or object from a pickle file."""
    if not os.path.exists(file_path):
        print(f"Error: The model file '{file_path}' does not exist.")
        return None
    with open(file_path, 'rb') as file:
        return joblib.load(file)

def test_model_with_dataset(file_path, dataset):
    """Loads a model and performs POS tagging using dataset sentences."""
    print(f"\nLoading model from {file_path}...")
    model = load_model(file_path)

    # Debug: Ensure model is loaded properly
    if model is None:
        print(f"Error: Could not load model from {file_path}")
        return

    # Debug: Check if the model has the required method
    if not hasattr(model, 'pos_tag_sentences'):
        print(f"Error: The model does not have the method 'pos_tag_sentences'.")
        return

    # Extract sentences from dataset and strip unnecessary characters (assuming '|')
    test_sentences = dataset['pos_tagged'].apply(lambda x: x.replace('|', '').strip()).tolist()

    print(f"Running POS tagging on {len(test_sentences)} test sentences using {file_path}...")

    try:
        tagged_sentences = model.pos_tag_sentences(test_sentences)
    except TypeError as e:
        print(f"An error occurred during POS tagging: {e}")
        return

    # Check if tagged_sentences is None or not a list
    if tagged_sentences is None:
        print(f"Error: 'pos_tag_sentences' returned None!")
        return
    if not isinstance(tagged_sentences, list):
        print(f"Error: 'pos_tag_sentences' did not return a list, but {type(tagged_sentences)}")
        return

    # Display the first 5 tagged sentences for review
    print("Sample Tagged Sentences (First 5):")
    for sentence in tagged_sentences[:5]:
        print(f"Tagged: {sentence}")

    # Optional: compare with expected labels from dataset
    print("\nComparison with expected labels (First 5):")
    for idx, (tagged, expected) in enumerate(zip(tagged_sentences[:5], dataset['label'][:5])):
        print(f"Tagged: {tagged} | Expected Label: {expected}")

def main():
    # Load the datasets
    bikol_dataset = pd.read_csv('../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/DATASETOFREGEX/Tagged_bikol.csv')
    cebuano_dataset = pd.read_csv('../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/DATASETOFREGEX/Tagged_cebuano.csv')
    tagalog_dataset = pd.read_csv('../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/DATASETOFREGEX/Tagged_tagalog.csv')

    # Paths to the saved models
    cebuano_model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/models/tagged_cebuano_model.pkl"
    bikol_model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/models/tagged_bikol_model.pkl"
    tagalog_model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/models/pos_tagalog_model.pkl"

    # Test Cebuano POS tagging with dataset
    print("Testing Cebuano POS Tagger with Dataset:")
    test_model_with_dataset(cebuano_model_path, cebuano_dataset)

    # Test Bikol POS tagging with dataset
    print("\nTesting Bikol POS Tagger with Dataset:")
    test_model_with_dataset(bikol_model_path, bikol_dataset)

    # Test Tagalog POS tagging with dataset
    print("\nTesting Tagalog POS Tagger with Dataset:")
    test_model_with_dataset(tagalog_model_path, tagalog_dataset)

if __name__ == "__main__":
    main()
