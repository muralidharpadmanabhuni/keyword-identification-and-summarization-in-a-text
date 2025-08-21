import os
from datasets import load_from_disk, DatasetDict
import re

# Define the paths
current_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
combined_dataset_path = os.path.join(current_dir, "xlsum_combined12")  # Path to combined dataset
processed_dataset_path = os.path.join(current_dir, "xlsum_processed")  # Path to save processed dataset

# Load the combined XLSum dataset
print(f"Loading combined dataset from: {combined_dataset_path}")
try:
    combined_dataset = load_from_disk(combined_dataset_path)
    print(f"Loaded dataset size: {len(combined_dataset)}")
except FileNotFoundError:
    print(f"Error: Combined dataset not found at {combined_dataset_path}. Please check the path.")
    exit(1)

# Function to clean text
def clean_data(text):
    """
    Cleans and normalizes the input text:
    - Removes unwanted characters (e.g., HTML tags, extra spaces, special symbols).
    - Normalizes whitespace.
    - Handles missing or None values.
    """
    if isinstance(text, list):  # If the text is a list, join the elements into one string
        text = " ".join(text)
    if text is None:
        return ""
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^\w\s.,!?;:']+", " ", text)  # Remove special characters except basic punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text

# Preprocess function for summarization (now handling batches)
def preprocess_function(examples):
    """
    Prepares the dataset for summarization:
    - Cleans `text` and `summary`.
    - Maps `text` to `input_text` (document to summarize).
    - Maps `summary` to `target_text` (reference summary).
    """
    # Clean the 'text' and 'summary' columns for each example
    input_texts = [clean_data(text) for text in examples["text"]]
    target_texts = [clean_data(summary) for summary in examples["summary"]]
    
    return {
        "input_text": input_texts,  # List of cleaned input texts
        "target_text": target_texts,  # List of cleaned summaries
    }

# Apply preprocessing to the dataset (now batched)
print("Preprocessing and cleaning the dataset...")
processed_dataset = combined_dataset.map(
    preprocess_function, 
    remove_columns=combined_dataset.column_names, 
    batched=True  # Process in batches
)

# Split the dataset into train, validation, and test sets
print("Splitting dataset into train, validation, and test sets...")
# 80% train, 10% validation, 10% test split
train_test_split = processed_dataset.train_test_split(test_size=0.2, seed=42)  # 80% train, 20% temp (test + validation)
validation_test_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)  # Split temp into 50% validation, 50% test

# Create DatasetDict with train, validation, and test splits
final_dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": validation_test_split["train"],
    "test": validation_test_split["test"],
})

# Save the processed dataset
print(f"Saving processed dataset to: {processed_dataset_path}")
os.makedirs(processed_dataset_path, exist_ok=True)
final_dataset.save_to_disk(processed_dataset_path)
print("Processed dataset saved successfully.")

# Verify the sizes of the splits
print(f"Train set size: {len(final_dataset['train'])}")
print(f"Validation set size: {len(final_dataset['validation'])}")
print(f"Test set size: {len(final_dataset['test'])}")
