import logging
import pandas as pd
import torch
from transformers import RobertaForMaskedLM, Trainer, TrainingArguments, RobertaTokenizerFast
from datasets import load_dataset, Dataset
from ast import literal_eval  # Import literal_eval for safe conversion of string lists
from DataCollector import CustomDataCollatorForPOS  # Import your custom data collator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tokenization and saving to CSV per sentence
def tokenize_and_save_to_csv(train_file, tokenizer, output_csv, vocab_size):
    logging.info("Loading dataset...")

    # Load the CSV dataset
    dataset = load_dataset('csv', data_files={'train': train_file}, split='train')
    logging.info(f"Dataset loaded from {train_file}")

    logging.info("Tokenizing the dataset per sentence...")

    tokenized_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    # Tokenize each sentence individually
    for example in dataset:
        logging.info("Tokenizing POS sequences...")

        # Split sentences into words for pre-tokenized input
        general_tokens = example['general'].split()
        detailed_tokens = example['detailed'].split()

        # Tokenize general and detailed POS sequences with truncation and padding
        tokenized_general = tokenizer(
            general_tokens,
            truncation=True,
            padding='max_length',  # Ensure consistent padding
            max_length=514,        # Set max_length to 514 to match Roberta's architecture
            is_split_into_words=True,
            return_tensors='pt'     # Return PyTorch tensors
        )
        tokenized_detailed = tokenizer(
            detailed_tokens,
            truncation=True,
            padding='max_length',  # Ensure consistent padding
            max_length=514,        # Set max_length to 514 to match Roberta's architecture
            is_split_into_words=True,
            return_tensors='pt'     # Return PyTorch tensors
        )

        # Check if token IDs are within the valid range of the model's vocabulary
        for ids in tokenized_general['input_ids']:
            if torch.any(ids >= vocab_size):
                logging.error(f"Found out-of-range token ID: {ids}")
                raise ValueError(f"Token ID out of range for model vocabulary: {ids.tolist()}")

        # Convert tensors to lists to store in CSV-compatible format
        tokenized_data["input_ids"].append(tokenized_general["input_ids"].tolist()[0])
        tokenized_data["attention_mask"].append(tokenized_general["attention_mask"].tolist()[0])
        tokenized_data["labels"].append(tokenized_detailed["input_ids"].tolist()[0])

    logging.info("Saving tokenized data to CSV...")

    # Convert tokenized data to DataFrame
    df_tokenized = pd.DataFrame({
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": tokenized_data["labels"]
    })

    # Write tokenized data to CSV
    df_tokenized.to_csv(output_csv, index=False)
    logging.info(f"Tokenized data saved to {output_csv}")

# Load tokenized data from CSV and convert to lists
def load_tokenized_data_from_csv(csv_file):
    logging.info(f"Loading tokenized data from {csv_file}")
    
    # Load tokenized data from CSV
    df_tokenized = pd.read_csv(csv_file)

    # Convert string representations back to Python lists (leave as lists for now)
    df_tokenized["input_ids"] = df_tokenized["input_ids"].apply(literal_eval)
    df_tokenized["attention_mask"] = df_tokenized["attention_mask"].apply(literal_eval)
    df_tokenized["labels"] = df_tokenized["labels"].apply(literal_eval)

    # Convert DataFrame to HuggingFace Dataset (lists are acceptable here)
    dataset = Dataset.from_pandas(df_tokenized)
    return dataset

# Convert lists back to tensors during training
def convert_lists_to_tensors(dataset):
    # Convert lists back to PyTorch tensors for model input
    dataset = dataset.map(lambda x: {
        'input_ids': torch.tensor(x['input_ids']),
        'attention_mask': torch.tensor(x['attention_mask']),
        'labels': torch.tensor(x['labels'])
    }, batched=True)

    return dataset

# Main function for training
def train_model_with_pos_tags(train_file, tokenizer, model, output_csv):
    # Get model's vocabulary size
    vocab_size = model.config.vocab_size
    logging.info(f"Model's vocabulary size before resizing: {vocab_size}")

    # Resize model token embeddings to match the tokenizer
    logging.info("Resizing model embeddings to match tokenizer vocabulary size...")
    model.resize_token_embeddings(len(tokenizer))

    vocab_size = len(tokenizer)
    logging.info(f"Model's vocabulary size after resizing: {vocab_size}")

    # Tokenize and save to CSV first
    tokenize_and_save_to_csv(train_file, tokenizer, output_csv, vocab_size)

    # Load the tokenized dataset from CSV (loaded as lists)
    tokenized_dataset = load_tokenized_data_from_csv(output_csv)
    logging.info("Tokenized dataset loaded successfully.")

    # Convert lists back to tensors
    tokenized_dataset = convert_lists_to_tensors(tokenized_dataset)

    # Set up the custom data collator for masked language modeling
    logging.info("Setting up data collator for masked language modeling...")
    data_collator = CustomDataCollatorForPOS(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Define training arguments
    logging.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        remove_unused_columns=False  # Prevent column removal
    )

    # Initialize Trainer
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # Train the model
    logging.info("Starting model training...")
    trainer.train()
    logging.info("Training completed successfully.")

# Load the tokenizer
logging.info("Loading the tokenizer...")
pos_tokenizer = RobertaTokenizerFast.from_pretrained(
    "jcblaise/roberta-tagalog-base",  # Use the same tokenizer as the model
    truncation=True,
    padding="max_length",
    max_length=514,
    add_prefix_space=True  # This ensures the tokenizer works with pretokenized inputs
)
logging.info("Tokenizer loaded successfully.")
