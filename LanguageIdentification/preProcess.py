import os
import re
import csv

class TextPreprocessor:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/dataset/dataset_{language}.csv"
        self.output_file = f"{results_folder}/preprocessed/preprocessed_{language}.csv"
        self.dictionary_file = f"{base_path}/LanguageIdentification/Dictionary/{language}_dictionary.csv"
        
        # Optional: Define noise words if needed
        self.noise_words = set([
            "na", "nang", "ng", "mga", "ang", "kung", "yan", "yun", "ayan", "sina", "sila",
            "baka", "ano", "anong", "mag", "doon", "si", "siya", "mo", "so", "ako", "ikaw",
            "po", "ko", "eme", "may", "luh", "ito", "ay", "ganon", "basta", "lang", "dito",
            "and", "i", "haha", "o", "pang", "daw", "raw", "aww", "kahit", "go", "rin", "din",
            "kayo", "hoy", "ok", "okay", "yung", "yay", "sa", "sabi", "eh", "sana", "da", 
            "ngani", "tabi", "ning", "kamo", "ini", "iyo", "sin", "kaya", "hali", "bala", 
            "aba", "alin", "ganiyan", "gaya", "ho", "ika", "kay", "kumusta", "naman", 
            "sapagkat", "tayo", "talaga", "wag", "yata", "ba", "bitaw", "dayon", "gani", 
            "kana", "mao", "diay", "mao ni", "mao ba", "usa", "kita", "kita tanan", "kamo", 
            "lagi", "gyud", "pud", "ahh", "pag", "pwede", "pwes", "pano", "ug"
        ])
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.dictionary_file), exist_ok=True)

    def preprocess_text(self, text):
        """Clean and lowercase text, removing special characters."""
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        return text

    def split_into_sentences(self, text):
        """Split text into sentences of 4 or more words."""
        return [chunk.strip() for chunk in re.split(r'[.!?]', text) if len(chunk.split()) >= 4]

    def process_file(self):
        """Preprocess text and save sentences into CSV."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                lines = infile.readlines()
                writer = csv.writer(outfile)

                for line in lines:
                    preprocessed_line = self.preprocess_text(line)
                    sentences = self.split_into_sentences(preprocessed_line)

                    for sentence in sentences:
                        writer.writerow([sentence])
            print(f"Preprocessed sentences saved at {self.output_file}")

        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def create_dictionary(self):
        """Create a dictionary of unique words without frequencies."""
        try:
            # Use self.output_file since this is the preprocessed file
            if not os.path.exists(self.output_file):
                print(f"Error: The file {self.output_file} does not exist.")
                return

            unique_words = set()

            with open(self.output_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                for row in reader:
                    if row:
                        text = row[0]
                        words = text.split()
                        # Exclude noise words
                        words = [word for word in words if word not in self.noise_words]
                        unique_words.update(words)

            if unique_words:
                with open(self.dictionary_file, 'w', newline='', encoding='utf-8') as dict_file:
                    writer = csv.writer(dict_file)
                    for word in sorted(unique_words):
                        writer.writerow([word])
                print(f"Dictionary saved at {self.dictionary_file}")
            else:
                print(f"No words found to create dictionary for {self.dictionary_file}")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    languages = ['tagalog', 'bikol', 'cebuano']

    for language in languages:
        processor = TextPreprocessor(language)
        processor.process_file()  # Preprocess the file
        processor.create_dictionary()  # Create the dictionary
