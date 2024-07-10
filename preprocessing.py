import re
import string

def clean_text(text):
    # Tokenization: 
    words = text.split()
    
    # Case Folding
    words = [word.lower() for word in words]
    
    # Data Cleaning
    words = [re.sub(f"[{re.escape(string.punctuation)}]", "", word) for word in words]
    
    # Remove empty strings 
    words = [word for word in words if word]
    
    # Join words back into a single string
    cleaned_text = " ".join(words)
    
    return cleaned_text

user_input = input("Type Here: ")

# Clean the input text
cleaned_text = clean_text(user_input)

# Output 
print("Cleaned text:", cleaned_text)
