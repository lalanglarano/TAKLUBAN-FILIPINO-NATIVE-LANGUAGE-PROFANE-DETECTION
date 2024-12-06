import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from LanguageIdentification.FNLI import LanguageIdentification
from PATTERN_GENERATION.tagalog import PatternGenerator as TagalogPatternGenerator
from PATTERN_GENERATION.bikol import PatternGenerator as BikolPatternGenerator
from PATTERN_GENERATION.cebuano import PatternGenerator as CebuanoPatternGenerator
import re

# Define noise words
noise_words = {
    "na", "nang", "ng", "mga", "ang", "kung", "yan", "ito", "si", "ko", "po", "ka", "ikaw", "siya", "oo",
    "sa", "may", "ni", "dahil", "kasi", "pero", "at", "para", "niya", "saan", "ganito", "doon", "noon", 
    "ta", "ngani", "ini", "kang", "iyo", "hali", "baga", "ho", "mo", "ba", "si", "kan", "kun", "ngani", "talagang",
    "yan", "sadi", "pala", "yaon", "ini", "yan", "na", "digdi", "dakol", "bangan", "dayon", "ang", "ini",
    "gani", "kana", "mao", "pud", "bitaw", "ta", "si", "ug", "naa", "dili", "kini", "adto", "man", "kay",
    "unta", "nga", "sa", "kani", "mo", "lang", "sila", "unsa", "ako", "niyo", "su", "talaga", "yung"
}

# Define function to load pattern generator
def get_pattern_generator(language):
    predefined_rules_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/PATTERN_GENERATION/predefined_rules.csv"
    model_filename = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'
    if language == 'tagalog':
        return TagalogPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    elif language == 'bikol':
        return BikolPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    elif language == 'cebuano':
        return CebuanoPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
    return None

# Define the profanity detection function
def predict_and_censor(sentence, pattern_generator, model):
    tokens = sentence.split()
    filtered_tokens = [token for token in tokens if token.lower() not in noise_words]
    if not filtered_tokens:
        return sentence, False

    censored_tokens = []
    is_profane = False
    for token in tokens:
        if token.lower() in noise_words:
            censored_tokens.append(token)
        else:
            if model.predict([token])[0] == 1:
                censored_tokens.append(re.sub(r'\w', '*', token))
                is_profane = True
            else:
                censored_tokens.append(token)

    censored_sentence = ' '.join(censored_tokens)
    return censored_sentence, is_profane

# Load models and test data
def evaluate_model(test_file, model_path, svm_model_path):
    print("Loading models...")
    model = joblib.load(model_path)
    language_identifier = LanguageIdentification(model=model, X_test=None, y_test=None)
    svm_model = joblib.load(svm_model_path)
    supported_languages = {'tagalog', 'bikol', 'cebuano'}
    
    test_data = pd.ExcelFile(test_file)
    overall_predictions = []
    overall_true_labels = []

    for sheet_name in test_data.sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")
        sheet_data = test_data.parse(sheet_name)
        sentences = sheet_data['sentence']
        true_labels = sheet_data['profane']
        
        predictions = []
        for sentence in sentences:
            predicted_language = language_identifier.predict_language(sentence)
            if predicted_language not in supported_languages:
                predictions.append(0)  # Assume non-profane for unsupported languages
                continue
            
            pattern_generator = get_pattern_generator(predicted_language)
            if pattern_generator:
                _, is_profane = predict_and_censor(sentence, pattern_generator, svm_model)
                predictions.append(1 if is_profane else 0)
            else:
                predictions.append(0)
        
        overall_predictions.extend(predictions)
        overall_true_labels.extend(true_labels)

        # Calculate and display metrics for each language
        print(f"\nClassification Report for {sheet_name}:")
        print(classification_report(true_labels, predictions))

        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['Not Profane', 'Profane'], yticklabels=['Not Profane', 'Profane'])
        plt.title(f"Confusion Matrix for {sheet_name}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    # Overall metrics across all sheets
    print("\nOverall Classification Report:")
    print(classification_report(overall_true_labels, overall_predictions))
    cm = confusion_matrix(overall_true_labels, overall_predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Not Profane', 'Profane'], yticklabels=['Not Profane', 'Profane'])
    plt.title("Overall Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    test_file = "testdata/Testing_Data.xlsx"
    model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"
    svm_model_path = "trained_profane_model.pkl"
    evaluate_model(test_file, model_path, svm_model_path)
