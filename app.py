from flask import Flask, render_template, request, jsonify
from TAKLUBAN import process_sentence, load_or_train_model  # Import the necessary functions from TAKLUBAN

app = Flask(__name__)

# Load the language identifier model once at the start of the app
language_identifier = load_or_train_model()

# Route for the home page
@app.route('/')
def home():
    return render_template('Home.html')

# Route for the creators page
@app.route('/creators')
def creators():
    return render_template('Creators.html')

# Route for the FAQs page
@app.route('/faqs')
def faqs():
    return render_template('FAQs.html')

# Route for detecting language and checking profanity
@app.route('/detect_language', methods=['POST'])
def detect_language():
    sentence = request.form['text']  # Get input text from HTML form
    if sentence:
        # Call the function from TAKLUBAN.py to process the sentence, passing both sentence and language_identifier
        predicted_language, pos_tagged_sentence, censored_sentence, is_profane = process_sentence(sentence, language_identifier)
        
        # Return the predicted language, POS tagged sentence, censored sentence, and profanity status as JSON response
        return jsonify({
            'predicted_language': predicted_language,
            'pos_tagged_sentence': pos_tagged_sentence if pos_tagged_sentence else 'POS tagging not performed',
            'censored_sentence': censored_sentence,
            'is_profane': is_profane
        })
    else:
        return jsonify({'error': 'No input provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
