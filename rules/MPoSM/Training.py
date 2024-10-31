import logging
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from ast import literal_eval
from DataCollector import CustomDataCollatorForPOS  # Import your custom data collator

# Configure logging to save to a file and print to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),  # Logs to file
        logging.StreamHandler()               # Prints to stdout
    ]
)

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

        # Ensure general and detailed POS tags are lists of tokens
        general_tokens = example['General POS'].split() if isinstance(example['General POS'], str) else example['General POS']
        detailed_tokens = example['Detailed POS'].split() if isinstance(example['Detailed POS'], str) else example['Detailed POS']

        # Tokenize general and detailed POS sequences with truncation and padding
        tokenized_general = tokenizer(
            general_tokens,
            truncation=True,
            padding='max_length',
            max_length=514,
            is_split_into_words=True,
            return_tensors='pt'
        )
        tokenized_detailed = tokenizer(
            detailed_tokens,
            truncation=True,
            padding='max_length',
            max_length=514,
            is_split_into_words=True,
            return_tensors='pt'
        )

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

    # Convert string representations back to Python lists
    df_tokenized["input_ids"] = df_tokenized["input_ids"].apply(literal_eval)
    df_tokenized["attention_mask"] = df_tokenized["attention_mask"].apply(literal_eval)
    df_tokenized["labels"] = df_tokenized["labels"].apply(literal_eval)

    # Convert DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df_tokenized)
    return dataset

# Convert lists back to tensors during training
def convert_lists_to_tensors(dataset):
    dataset = dataset.map(lambda x: {
        'input_ids': torch.tensor(x['input_ids']),
        'attention_mask': torch.tensor(x['attention_mask']),
        'labels': torch.tensor(x['labels'])
    }, batched=True)
    return dataset

def train_model_with_pos_tags(train_file, tokenizer, model, output_csv, resume_from_checkpoint=None):
    vocab_size = model.config.vocab_size
    logging.info(f"Model's vocabulary size before resizing: {vocab_size}")

    # Resize model token embeddings to match the tokenizer
    logging.info("Resizing model embeddings to match tokenizer vocabulary size...")
    model.resize_token_embeddings(len(tokenizer))

    vocab_size = len(tokenizer)
    logging.info(f"Model's vocabulary size after resizing: {vocab_size}")

    # Tokenize and save to CSV first
    tokenize_and_save_to_csv(train_file, tokenizer, output_csv, vocab_size)

    # Load the tokenized dataset from CSV
    dataset = load_tokenized_data_from_csv(output_csv)
    logging.info("Tokenized dataset loaded successfully.")

    # Split the dataset for training and evaluation
    dataset_split = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    # Convert lists back to tensors
    train_dataset = convert_lists_to_tensors(train_dataset)
    eval_dataset = convert_lists_to_tensors(eval_dataset)  # Ensure eval_dataset is converted as well

    # Set up the custom data collator for MLM
    logging.info("Setting up data collator for masked language modeling...")
    data_collator = CustomDataCollatorForPOS(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Define training arguments with checkpointing settings
    logging.info("Setting up training arguments with checkpoint saving...")
    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./results/logs",
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=100,
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        remove_unused_columns=False,
        save_total_limit=3
    )

    # Initialize Trainer with eval_dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Provide eval_dataset for evaluation
        data_collator=data_collator
    )

    # Resume training from checkpoint if available
    logging.info("Starting model training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logging.info("Training completed successfully.")

    # Save final model and tokenizer
    model.save_pretrained("./results/final_model")
    tokenizer.save_pretrained("./results/final_tokenizer")
    logging.info("Model and tokenizer saved to ./results.")


