import logging
from transformers import AutoTokenizer, RobertaForMaskedLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize a tokenizer from a pre-trained model
pos_tokenizer = AutoTokenizer.from_pretrained('jcblaise/roberta-tagalog-base', add_prefix_space=True)
logging.info("Initializing Roberta Tokenizer...")

# Initialize the model
model = RobertaForMaskedLM.from_pretrained("jcblaise/roberta-tagalog-base")
logging.info("Initializing Roberta Model...")

# Vocabulary for POS tags (both general and detailed)
logging.info("Setting up POS tag vocabulary...")
pos_tag_vocab = [
    "NNC", "NNP", "NNPA", "NNCA", "PRS", "PRP", "PRSP", "PRO", "PRQ", "PRQP", "PRL", "PRC", 
    "PRF", "PRI", "DTC", "DTCP", "DTP", "DTPP", "CCT", "CCR", "CCB", "CCA", "CCP", "CCU", 
    "VBW", "VBS", "VBH", "VBN", "VBTS", "VBTR", "VBTF", "VBTP", "VBAF", "VBOF", "VBOB", 
    "VBOL", "VBOI", "VBRF", "JJD", "JJC", "JJCC", "JJCS", "JJCN", "JJN", "RBD", "RBN", "RBK", 
    "RBP", "RBB", "RBR", "RBQ", "RBT", "RBF", "RBW", "RBM", "RBL", "RBI", "RBJ", "RBS", "CDB", 
    "PMP", "PME", "PMQ", "PMC", "PMSC", "PMS", "LM", "TS", "FW", "NN.*", "PR.*", "DT.*", "VB.*", 
    "CC.*", "JJ.*", "RB.*", "CD.*", "PM.*", "[MASK]", "[PAD]", "[UNK]"
]

# Add POS tag vocabulary
logging.info("Adding vocabulary to the tokenizer...")
pos_tokenizer.add_tokens(pos_tag_vocab)

# Save tokenizer for later use
logging.info("Saving tokenizer...")
pos_tokenizer.save_pretrained("model/pos_tokenizer")
logging.info("Tokenizer saved successfully.")

# Load pre-trained model
logging.info("Loading pre-trained model...")
model = RobertaForMaskedLM.from_pretrained("jcblaise/roberta-tagalog-base")

# Resize token embeddings
logging.info("Resizing model embeddings...")
model.resize_token_embeddings(len(pos_tokenizer))

# Save the resized model
logging.info("Saving the resized model...")
model.save_pretrained("model/pos_model")
logging.info("Resized model saved successfully.")

print(len(pos_tokenizer))
