import os
from datasets import load_dataset

# Directory to save processed data
os.makedirs('data/finqa_combined', exist_ok=True)

# Load the dataset splits
dataset = load_dataset('llk010502/FinQA_Combined_dataset')

# Function to select only the required columns
def select_columns(example):
    return {
        'question': example['question'],
        'answer': example['answer'],
        'information': example['information'],  # Use directly
    }

# Process and save each split
for split in ['train', 'validation', 'test']:
    if split in dataset:
        data = dataset[split].map(select_columns)
        data.to_parquet(f'data/finqa_combined/{split}.parquet')
        print(f"Saved {split} split to data/finqa_combined/{split}.parquet") 