import pickle
import nltk

# Load the POS Tagging model
model_path = '../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION/Results/models/pos_tagalog_model.pkl'

# Function to load the pre-trained model
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to tokenize and predict POS tags
def pos_tagging(sentence, model):
    # Tokenize the input sentence
    tokens = nltk.word_tokenize(sentence)
    
    # Predict POS tags
    tagged_sentence = model.predict([tokens])
    
    # Display the result
    for word, tag in zip(tokens, tagged_sentence[0]):
        print(f"{word} | {tag}")

# Load the pre-trained model
pos_model = load_model(model_path)

# Ask the user for input
user_sentence = input("Enter a sentence: ")

# Output POS tags for the sentence
pos_tagging(user_sentence, pos_model)
