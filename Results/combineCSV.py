import pandas as pd

# List of CSV files to combine
csv_files = ['Results\dataset\dataset_bikol.csv', 'Results\dataset\dataset_cebuano.csv', 'Results\dataset\dataset_tagalog.csv']

# Read and concatenate all CSV files
combined_df = pd.concat([pd.read_csv(file) for file in csv_files])

# Write the combined data to a new CSV file
output_file = 'Results\combined_output.csv'
combined_df.to_csv(output_file, index=False)

print(f"Combined CSV file has been created: '{output_file}'")