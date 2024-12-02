"""
Program Title: Profanity Detection Web Application

Programmers: Jeo Abarre, Telish Gonzales

Where the program fits in the general system designs: This module provides a web interface for detecting and censoring profane language in various Filipino languages.

Date written and last revised: October 5, 2024   |   November 18, 2024

Purpose: To provide a web interface for detecting and censoring profane language in 
Tagalog, Bikol, and Cebuano. This module uses Flask to create a web application that allows 
users to input text, detects the language, tags the text with POS tags, and censors any detected 
profane patterns.

Data structures, algorithms, and control:

Data Structures:
pattern_generators: A dictionary of pattern generators for each language.
language_identifier: The trained language identification model.
svm_model: The pre-trained profanity detection model.

Algorithms:
Uses Flask for the web interface.
Uses joblib to load pre-trained models.

Control:
Initializes pattern generators and loads models.
Defines routes for various web pages.
Detects language and censors profane patterns in user input.
Returns results as JSON responses.
"""

from flask import Flask, render_template, request, jsonify
from TAKLUBAN import predict_and_censor, get_pattern_generator
from LanguageIdentification.FNLI import LanguageIdentification, ModelTraining
import joblib

app = Flask(__name__)

# Paths and initialization
model_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/saved_model.pkl"
dictionary_dir = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/LanguageIdentification/Dictionary"
svm_model_path = "trained_profane_model.pkl"

# Global variables for models
language_model = None
X_test, y_test = None, None
svm_model = None

def load_models():
    """Function to load models during app startup."""
    global language_model, X_test, y_test, svm_model

    # Load or train the language identification model
    if not language_model:
        trainer = ModelTraining(dictionary_dir)
        language_model, X_test, y_test = trainer.train_model()
        joblib.dump(language_model, model_path)
    else:
        language_model = joblib.load(model_path)

    # Load SVM model for profanity detection
    if not svm_model:
        svm_model = joblib.load(svm_model_path)

# Load models during app startup
with app.app_context():
    load_models()

# Routes for various pages
@app.route('/')
def home():
    """Render the Home page."""
    return render_template('Home.html')

@app.route('/index')
def index():
    """Render the Index page."""
    return render_template('index.html')

@app.route('/creators')
def creators():
    """Render the Creators page."""
    return render_template('Creators.html')

@app.route('/faqs')
def faqs():
    """Render the FAQs page."""
    return render_template('FAQs.html')

@app.route('/detect_language', methods=['POST'])
def detect_language():
    """API endpoint to detect language, perform POS tagging, and censor profane words."""
    text = request.form.get('text', '').strip()
    if not text:
        return jsonify({
            'error': 'No input text provided'
        })

    # Predict the language
    language_identifier = LanguageIdentification(model=language_model, X_test=X_test, y_test=y_test)
    predicted_language = language_identifier.predict_language(text)
    supported_languages = {'tagalog', 'bikol', 'cebuano'}

    if predicted_language not in supported_languages:
        return jsonify({
            'predicted_language': predicted_language,
            'error': f"The language '{predicted_language}' is not supported."
        })

    # Get the pattern generator for the predicted language
    pattern_generator = get_pattern_generator(predicted_language)
    if not pattern_generator:
        return jsonify({
            'predicted_language': predicted_language,
            'error': f"Pattern generator for '{predicted_language}' is not available."
        })

    # Perform POS tagging and censorship
    pos_tagged_sentence = ' '.join(pattern_generator.tag_sentence(text))
    censored_sentence, is_profane = predict_and_censor(text, pattern_generator, svm_model, predicted_language)

    return jsonify({
        'predicted_language': predicted_language,
        'pos_tagged_sentence': pos_tagged_sentence,
        'censored_sentence': censored_sentence,
        'is_profane': is_profane
    })

if __name__ == '__main__':
    app.run(debug=True)
