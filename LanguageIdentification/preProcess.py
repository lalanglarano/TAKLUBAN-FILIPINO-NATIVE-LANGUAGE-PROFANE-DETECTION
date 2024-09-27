import os
import csv
import re

class TextPreprocessor:
    def __init__(self, language):
        base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
        results_folder = f"{base_path}/Results"
        self.input_file = f"{results_folder}/dataset/dataset_{language}.csv"
        self.output_dir = f"{results_folder}/preprocessed/"
        self.output_file = f"{self.output_dir}/preprocessed_{language}.csv"
        self.dictionary_dir = f"{base_path}/LanguageIdentification/Dictionary/"
        self.dictionary_file = f"{self.dictionary_dir}/{language}_dictionary.csv"
        self.noise_words = set([
            "na", "nang", "ng", "mga", "ang", "kung", "yan", "yun", "ayan", "sina", "sila",
            "baka", "ano", "anong", "mag", "doon", "siya", "mo", "so", "ako",
            "po", "ko", "eme", "may", "luh", "ito", "ay", "ganon", "basta", "lang",
            "and", "i", "haha", "o", "pang", "daw", "raw", "aww", "kahit", "go", "rin", "din",
            "kayo", "baka", "hoy", "ok", "okay", "yung", "yay", "sa", "sabi", "eh", "sana",
            "da", "ngani", "tabi", "ning", "kamo", "ini", "iyo", "sin", "kaya", "basta",
            "hali", "bala", "aba", "alin", "baka", "baga", "ganiyan", "gaya", "ho", "ika",
            "kay", "mo", "naman", "po", "sapagkat", "tayo", "talaga", "wag",
            "naman", "yata", "ba", "bitaw", "dayon", "gani", "kana", "mao", "diay", "mao ni",
            "mao ba", "lang", "usa", "kita", "kita tanan", "kamo", "ta", "lagi", "gyud",
            "bitaw", "pud", "ahh", "pag", "pwede", "pwes", "pano", "ok", "ug"
        ])
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dictionary_dir, exist_ok=True)

    def preprocess_text(self, text):
        # Remove starting double quotes in the sentence
        text = text.lstrip('"')

        # Convert to lowercase
        text = text.lower()

        # Remove non-alphanumeric characters except spaces and commas
        text = ''.join(char if char.isalnum() or char in [' ', ','] else '' for char in text)

        # Remove digits
        text = ''.join(char if not char.isdigit() else '' for char in text)

        # Filter out noise words
        text = ' '.join(word for word in text.split() if word not in self.noise_words)

        return text

    def split_into_sentences(self, text):
        # Split the text based on punctuation marks: periods, exclamation marks, question marks, colons, and commas
        chunks = re.split(r'[.!?:]', text)

        sentences = []
        for chunk in chunks:
            # Split the chunk based on commas and filter out segments with fewer than 2 words
            sub_sentences = re.split(r',', chunk)
            for sub in sub_sentences:
                words = sub.strip().split()
                if len(words) >= 2:  # Accept only if 2 or more consecutive words
                    sentences.append(sub.strip())

        return sentences

    def preprocess_csv(self):
        try:
            if not os.path.exists(self.input_file):
                print(f"Error: The file {self.input_file} does not exist.")
                return
            
            word_count = {}
            with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                lines = infile.readlines()
                writer = csv.writer(outfile)

                for line in lines:
                    preprocessed_line = self.preprocess_text(line)
                    sentences = self.split_into_sentences(preprocessed_line)

                    for sentence in sentences:
                        writer.writerow([sentence])

                        # Update the word count dictionary
                        words = sentence.split()
                        for word in words:
                            if not word.isnumeric():
                                word_count[word] = word_count.get(word, 0) + 1

            if word_count:
                with open(self.dictionary_file, 'w', newline='', encoding='utf-8') as dict_file:
                    writer = csv.writer(dict_file)
                    writer.writerow(['word', 'frequency'])
                    for word, freq in sorted(word_count.items()):
                        writer.writerow([word, freq])
                print(f"Dictionary saved at {self.dictionary_file}")
            else:
                print(f"No words found after preprocessing for {self.dictionary_file}")

        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
