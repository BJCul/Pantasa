import logging
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from Training import train_model_with_pos_tags  # Import your training function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Starting the process...")

    # Load custom tokenizer (with the POS tag vocabulary added)
    logging.info("Loading the custom tokenizer...")
    pos_tokenizer = RobertaTokenizerFast.from_pretrained(
        "/model/pos_tokenizer",  # Path to your custom tokenizer with POS tags
        truncation=True,
        padding="max_length",
        max_length=1000,
        add_prefix_space=True  # Ensures the tokenizer works with pretokenized inputs
    )
    logging.info("Tokenizer loaded successfully.")

    # Load the model
    logging.info("Loading the pre-trained model...")
    model = RobertaForMaskedLM.from_pretrained("jcblaise/roberta-tagalog-base")  # Load the base model
    logging.info("Model loaded successfully.")

    # Resize the model's token embeddings to match the tokenizer's vocabulary size
    logging.info("Resizing model token embeddings to match tokenizer vocabulary...")
    model.resize_token_embeddings(len(pos_tokenizer))  # Resize embeddings based on tokenizer's vocab size

    logging.info("Model loaded and resized successfully with custom tokens.")

    # Path to the CSV input file
    csv_input = "rules/MPoSM/pos_tags_output.csv"

    # Path to the output CSV where tokenized data will be saved
    output_csv = "rules/MPoSM/tokenized_output.csv"

    logging.info("Starting training with file: %s", csv_input)

    # Call to the training function, passing the output CSV path
    train_model_with_pos_tags(csv_input, pos_tokenizer, model, output_csv)

    logging.info("Training completed.")
