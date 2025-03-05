import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

# Example of re-labeling data (simplified)
sentences = ["napaka bobo talaga", "ang galing galing mo naman"]
labels = [1, 0]  # Sentence-level labels

# Tokenize and create word-level labels
word_labels = []
for sentence, label in zip(sentences, labels):
    words = sentence.split()
    word_labels.extend([(word, label) for word in words])

# Separate words and labels
words, word_labels = zip(*word_labels)

# Vectorize words using N-grams
vectorizer = CountVectorizer(ngram_range=(1, 2))  # Using bigrams
X = vectorizer.fit_transform(words)

# Train SVM model
svm_model = SVC()
svm_model.fit(X, word_labels)

# Function to censor sentence
def censor_sentence(sentence, model, vectorizer):
    tokens = sentence.split()
    censored_tokens = []
    
    for i in range(len(tokens)):
        # Create context window for N-grams
        context = ' '.join(tokens[max(0, i-1):i+1])  # Adjust window size as needed
        token_vector = vectorizer.transform([context])
        if model.predict(token_vector)[0] == 1:  # Assuming 1 indicates a profane word
            censored_tokens.append(re.sub(r'\w', '*', tokens[i]))
        else:
            censored_tokens.append(tokens[i])
    
    return ' '.join(censored_tokens)

# Example usage
sentence = "ang astig mo don kahit bobo kakampi mo"
censored_sentence = censor_sentence(sentence, svm_model, vectorizer)
print(censored_sentence)

# For bigrams, the model will not detect "bobo kakampi" as profane
# Checking of Annalyn, to integrate it on takluban
# Adding confusion matrix for overall accuracy