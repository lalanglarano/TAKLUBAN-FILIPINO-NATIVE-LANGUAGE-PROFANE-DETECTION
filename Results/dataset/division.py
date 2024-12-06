import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
dataset_path = 'Results/PFW/preprocessed_cebuano.csv'
df = pd.read_csv(dataset_path, header=None, names=['sentence', 'profane'])

# Split the dataset into training (80%) and remaining (20%)
train_df, remaining_df = train_test_split(df, test_size=0.6, random_state=42)

# Split the remaining dataset into validation (50% of remaining) and testing (50% of remaining)
val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)

# Save the splits into separate CSV files
train_df.to_csv('Results/dataset/cebuano/dataset_cebuano_train.csv', index=False, header=False)
val_df.to_csv('Results/dataset/cebuano/dataset_cebuano_val.csv', index=False, header=False)
test_df.to_csv('Results/dataset/cebuano/dataset_cebuano_test.csv', index=False, header=False)

# Print the sizes of the splits to verify
print(f"Total dataset size: {len(df)}")
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Testing set size: {len(test_df)}")

print("Datasets saved successfully.")