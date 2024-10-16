from flask import Flask, render_template, request, jsonify
from TAKLUBAN import process_sentence  # Import the process_sentence function from TAKLUBAN

app = Flask(__name__)

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

# Route for detecting language
@app.route('/detect_language', methods=['POST'])
def detect_language():
    sentence = request.form['text']  # Get input text from HTML form
    if sentence:
        # Call the function from TAKLUBAN.py to process the sentence
        predicted_language, pos_tagged_sentence = process_sentence(sentence)
        
        # Return the predicted language and POS tagged sentence as JSON response
        return jsonify({
            'predicted_language': predicted_language,
            'pos_tagged_sentence': pos_tagged_sentence if pos_tagged_sentence else 'POS tagging not performed'
        })
    else:
        return jsonify({'error': 'No input provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
