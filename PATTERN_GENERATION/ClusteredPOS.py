import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def parse_pos_data(pos_data):
    """Parses the POS data to separate words and their tags."""
    words = []
    pos_tags = []

    # Split the string by spaces to get each word/POS pair
    for entry in pos_data.split():
        if '|' in entry:
            # Split by '|' to separate the word from the POS tag
            word, pos_tag = entry.split('|', 1)  # Use `1` to limit splitting to only the first '|'
            words.append(word)
            pos_tags.append(pos_tag)
        else:
            print(f"Warning: Skipping malformed entry '{entry}'")
    
    return words, pos_tags

def cluster_pos_tags(language):
    # Set the base path and the input/output file paths
    base_path = "../TAKLUBAN-FILIPINO-NATIVE-LANGUAGE-PROFANE-DETECTION"
    results_folder = f"{base_path}/Results"
    input_file = f"{results_folder}/pos_tagged/FPOSTagged_{language}.csv"
    output_file = f"{results_folder}/CLUSTEREDTAG/Clustered_{language}.csv"

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: The file {input_file} does not exist.")
        return

    # Read the CSV file containing POS-tagged data
    df = pd.read_csv(input_file, header=None, names=['pos_data'])

    # Parse the POS data into words and POS tags
    words_list = []
    pos_tags_list = []

    for pos_data in df['pos_data']:
        words, pos_tags = parse_pos_data(pos_data)
        words_list.extend(words)
        pos_tags_list.extend(pos_tags)

    # Vectorize the POS tags using TF-IDF to prepare them for clustering
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(pos_tags_list)

    # Use KMeans to cluster the POS tags
    num_clusters = 5  # You can change the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Prepare the output DataFrame
    output_df = pd.DataFrame({
        'word': words_list,
        'pos_tag': pos_tags_list,
        'cluster': clusters
    })

    # Save the clustered data to a new CSV file
    output_df.to_csv(output_file, index=False)

    print(f"Clustering complete. Output saved to {output_file}.")

# List of languages to process
languages = ['tagalog', 'bikol', 'cebuano']

# Loop through each language and cluster POS tags
for language in languages:
    cluster_pos_tags(language)
