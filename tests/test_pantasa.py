from preprocessing_module import preprocess_text  # Import your preprocessing module
from rule_checker import process_input_ngram  # Import rule checking function

# Step 1: Input the text
input_text = "siya ay kumain nang mansanas"

# Step 2: Preprocess the input text (tokenize, POS-tag, lemmatize)
tokens, lemmas, pos_tags = preprocess_text(input_text)

# Step 3: Combine the preprocessed tokens and POS tags into n-grams (example: "NN VB DT" format)
ngram_input = ' '.join([f"{pos}" for word, pos in pos_tags])

print(f"Generated N-gram: {ngram_input}")

# Step 4: Pass the generated n-gram to the rule-checking module
corrections = process_input_ngram(ngram_input)

# Step 5: Output the corrections and suggestions
print("Corrections:")
for correction in corrections:
    print(f"Pattern ID: {correction['pattern_id']}, Distance: {correction['distance']}, Tags: {correction['correction_tags']}")
