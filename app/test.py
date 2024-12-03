# test_pantasa.py

import os
import logging

# Set up logging to display debug information
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import the necessary functions from your Pantasa module
# Replace 'pantasa_module' with the actual name of your module file (without the .py extension)
from pantasa_checker import *

# Set up the input and file paths
input_sentence = "Ang bata ay kumakain ng mansanas"
jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
rule_path = 'data/processed/detailed_hngram.csv'
directory_path = 'data/raw/dictionary.csv'
pos_path = 'data/processed/pos_dic'

# Ensure that all file paths exist
assert os.path.exists(jar_path), f"Jar file not found at {jar_path}"
assert os.path.exists(model_path), f"Model file not found at {model_path}"
assert os.path.exists(rule_path), f"Rule pattern bank not found at {rule_path}"
assert os.path.exists(directory_path), f"Dictionary file not found at {directory_path}"
assert os.path.exists(pos_path), f"POS dictionary path not found at {pos_path}"

# Step 1: Tokenize the input sentence
tokens = tokenize_sentence(input_sentence)
print("Tokens:", tokens)

# Step 2: POS tag the tokens
pos_tags = pos_tagging(tokens, jar_path=jar_path, model_path=model_path)
print("POS Tags:", pos_tags)

# Step 3: Load the rule pattern bank
rule_bank = rule_pattern_bank(rule_path)
print("Rule Pattern Bank Loaded.")

# Step 4: Generate suggestions using n-gram matching
token_suggestions = generate_suggestions(pos_tags, rule_path)
print("Token Suggestions:")
for idx, suggestion in enumerate(token_suggestions):
    print(f"Token {idx}: {suggestion}")

# Step 5: Apply POS corrections
corrected_sentence = apply_pos_corrections(token_suggestions, pos_tags, pos_path)
print("Corrected Sentence:", corrected_sentence)
