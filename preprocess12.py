import logging
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_and_split_data():
    # Load dataset
    dataset_path = "D:/project/wikiann_combined12"  # Adjust path as needed
    logger.info("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    logger.info(f"Loaded dataset: {dataset}")

    # Split dataset (if not already split)
    if not isinstance(dataset, DatasetDict):
        logger.info("No predefined splits found. Splitting dataset into train and test...")
        dataset_split = dataset.train_test_split(test_size=0.3)  # 30% test, 70% train
        dataset = DatasetDict({
            "train": dataset_split["train"],
            "test": dataset_split["test"]
        })
        logger.info("Dataset split into train and test sets.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    # Preprocess function
    def preprocess_data(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
            max_length=128  # Adjust based on your needs
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)  # Padding token label
                elif word_id != previous_word_id:
                    label_ids.append(label[word_id])  # Assign NER tag to word token
                else:
                    label_ids.append(-100)  # Avoid repetition
                previous_word_id = word_id
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Apply the preprocessing function
    logger.info("Preprocessing the dataset...")
    preprocessed_data = dataset.map(preprocess_data, batched=True)

    # Save preprocessed dataset
    preprocessed_data_path = "D:/project/ner_data123"
    preprocessed_data.save_to_disk(preprocessed_data_path)
    logger.info(f"Preprocessed dataset saved to: {preprocessed_data_path}")
    
    return preprocessed_data

# Run preprocessing and splitting
if __name__ == "__main__":
    preprocess_and_split_data()
