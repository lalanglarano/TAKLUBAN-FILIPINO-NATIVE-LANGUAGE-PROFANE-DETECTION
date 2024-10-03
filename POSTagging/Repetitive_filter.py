import pandas as pd

# Read the CSV file into a pandas DataFrame
input_file = 'Results/FilteredFW/bikol_repetitive_filtered.csv'  # Replace with the path to your CSV file
output_file = 'Results/FilteredFW/bikol(1)_repetitive_filtered.csv'  # Output file to store the filtered data

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file, header=None, names=['Word'])

# Remove duplicates
df_unique = df.drop_duplicates()

# Write the filtered data to a new CSV file
df_unique.to_csv(output_file, index=False, header=False)

print(f"Filtered data saved to {output_file}")
