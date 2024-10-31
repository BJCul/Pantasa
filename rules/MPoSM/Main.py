import logging
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from Training import train_model_with_pos_tags  # Import your training function
import os
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Starting the process...")

    # Load custom tokenizer (with the POS tag vocabulary added)
    logging.info("Loading the custom tokenizer...")
    pos_tokenizer = RobertaTokenizerFast.from_pretrained(
        "model/pos_tokenizer",  # Path to your custom tokenizer with POS tags
        truncation=True,
        padding="max_length",
        max_length=1000,
        add_prefix_space=True,
        use_cache=False
    )
    logging.info("Tokenizer loaded successfully.")

    # Load the pre-trained model
    logging.info("Loading the pre-trained model...")
    model = RobertaForMaskedLM.from_pretrained("jcblaise/roberta-tagalog-base")
    logging.info("Model loaded successfully.")

    # Resize the model's token embeddings to match the tokenizer's vocabulary size
    logging.info("Resizing model token embeddings to match tokenizer vocabulary...")
    model.resize_token_embeddings(len(pos_tokenizer))  # Resize embeddings based on tokenizer's vocab size

    # Wrap the model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Model loaded and resized successfully with custom tokens.")

    # Path to the CSV input file
    csv_input = "rules/MPoSM/pos_tags_output.csv"
    output_csv = "rules/MPoSM/tokenized_output.csv"

    # Check for the latest checkpoint
    checkpoint_dir = "./results"
    resume_from_checkpoint = None
    if os.path.isdir(checkpoint_dir):
        checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            resume_from_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
            logging.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

    # Call to the training function, passing the output CSV path and checkpoint
    train_model_with_pos_tags(csv_input, pos_tokenizer, model, output_csv, resume_from_checkpoint)

    logging.info("Training completed.")
