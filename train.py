import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk, ClassLabel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    preprocessed_data_path = "./ner_data123"
    model_checkpoint = "distilbert-base-multilingual-cased"
    output_dir = "./ner_model_output"

    logger.info("Loading preprocessed dataset...")
    dataset = load_from_disk(preprocessed_data_path)
    logger.info(f"Loaded dataset: {dataset}")

    if isinstance(dataset["train"].features["labels"].feature, ClassLabel):
        num_labels = dataset["train"].features["labels"].feature.num_classes
    else:
        unique_labels = set()
        for example in dataset["train"]["labels"]:
            unique_labels.update(example)
        num_labels = len(unique_labels)
    logger.info(f"Number of labels: {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}.")

if __name__ == "__main__":
    main()
