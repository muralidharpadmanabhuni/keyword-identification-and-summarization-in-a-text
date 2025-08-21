import os
import torch
from datasets import load_from_disk
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "xlsum_processed")
model_save_path = os.path.join(current_dir, "xlsum_model_output")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading and shuffling dataset...")
dataset = load_from_disk(dataset_path)
dataset = dataset.shuffle(seed=42)

model_name = "t5-small"
print(f"Loading model: {model_name}")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.gradient_checkpointing_enable()

def preprocess_data(batch):
    inputs = tokenizer(
        batch["input_text"],
        max_length=256,
        truncation=True,
        padding="max_length",
    )
    targets = tokenizer(
        batch["target_text"],
        max_length=150,
        truncation=True,
        padding="max_length",
    )
    batch["input_ids"] = inputs["input_ids"]
    batch["attention_mask"] = inputs["attention_mask"]
    batch["labels"] = targets["input_ids"]
    return batch

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=["input_text", "target_text"])

training_args = TrainingArguments(
    output_dir=model_save_path,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print(f"Saving trained model to: {model_save_path}")
trainer.save_model(model_save_path)
print("Training completed successfully!")
