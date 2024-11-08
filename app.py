from flask import Flask, render_template, request, jsonify
import joblib
import os
from TAKLUBAN import LanguageIdentification, TagalogPatternGenerator, BikolPatternGenerator, CebuanoPatternGenerator, predict_and_censor

app = Flask(__name__)

# Paths required for initializing pattern generators
predefined_rules_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/PATTERN_GENERATION/predefined_rules.csv"
model_filename = 'Modules/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
path_to_jar = 'Modules/FSPOST/stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'

# Load the language identifier model once at the start of the app
try:
    language_identifier = LanguageIdentification.load_or_train_model()
except Exception as e:
    print(f"Error loading language identifier model: {e}")
    language_identifier = None  # Set to None if loading fails

# Load the pre-trained profanity detection model
model_path = "trained_profane_model.pkl"
svm_model = joblib.load(model_path)

# Initialize pattern generators with required arguments
pattern_generators = {
    'tagalog': TagalogPatternGenerator(predefined_rules_path, model_filename, path_to_jar),
    'bikol': BikolPatternGenerator(predefined_rules_path, model_filename, path_to_jar),
    'cebuano': CebuanoPatternGenerator(predefined_rules_path, model_filename, path_to_jar)
}

# Routes for various pages
@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/creators')
def creators():
    return render_template('Creators.html')

@app.route('/faqs')
def faqs():
    return render_template('FAQs.html')

# Language detection and profanity filtering endpoint
@app.route('/detect_language', methods=['POST'])
def detect_language():
    text = request.form.get('text')
    
    # Ensure language_identifier is loaded
    if language_identifier is None:
        return jsonify({"error": "Language model not loaded"}), 500
    
    # Detect language
    predicted_language = language_identifier.predict_language(text)
    if predicted_language not in pattern_generators:
        return jsonify({"error": "Language not supported"}), 400
    
    # POS tagging and profanity check
    pattern_generator = pattern_generators[predicted_language]
    pos_tagged_sentence = pattern_generator.tag_sentence(text)
    censored_sentence, is_profane = predict_and_censor(text, pattern_generator, svm_model, predicted_language)

    response = {
        "predicted_language": predicted_language,
        "pos_tagged_sentence": " ".join(pos_tagged_sentence),
        "censored_sentence": censored_sentence,
        "is_profane": is_profane
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
