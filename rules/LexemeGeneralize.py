import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os

# Define the model
global_max_score = 0
global_min_score = 0
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"
print("Loading tokenizer...")
roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)
print("Loading model...")
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)
print("Model and tokenizer loaded successfully.")

# Function to read CSV in chunks of n rows
def load_csv_in_chunks(file_path, chunk_size=2):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        chunk = []
        for i, row in enumerate(reader, 1):
            chunk.append(row)
            if i % chunk_size == 0:
                print(f"Processing chunk {i // chunk_size}")
                yield chunk
                chunk = []
        if chunk:  # Yield the last chunk if it's less than chunk_size
            print("Processing the last chunk of the file")
            yield chunk

# Function to load an entire CSV as a list of dictionaries
def load_csv_as_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]  # Returns a list of dictionaries

# Helper function to load lexeme comparison dictionary
def load_lexeme_comparison_dictionary(file_path):
    comparisons = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split("::")
                if len(parts) == 3:
                    rough_pos, lexeme_ngram, pattern_id = parts
                    comparisons[(rough_pos, lexeme_ngram)] = pattern_id
                else:
                    print(f"Skipping malformed line: {line.strip()}")
    except FileNotFoundError:
        print(f"Lexeme comparison dictionary file not found: {file_path}")
    return comparisons

# Helper function to save lexeme comparison dictionary
def save_lexeme_comparison_dictionary(file_path, dictionary):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for key, value in dictionary.items():
                rough_pos, lexeme_ngram = key
                file.write(f"{rough_pos}::{lexeme_ngram}::{value}\n")
    except Exception as e:
        print(f"Error writing lexeme comparison dictionary: {e}")

# Helper function to convert ID_Array field to a list
def convert_id_array(id_array_str):
    if id_array_str is None:
        return []  # Return an empty list or handle it accordingly
    return id_array_str.strip("[]'").replace("'", "").split(', ')

# Function to load and convert ID_Array in CSV chunks
def load_and_convert_csv_in_chunks(file_path, chunk_size=2):
    for chunk in load_csv_in_chunks(file_path, chunk_size):
        for entry in chunk:
            entry['ID_Array'] = convert_id_array(entry.get('ID_Array', ''))
        yield chunk

# Compute MLM score for a given sentence
def compute_mlm_score(sentence, model, tokenizer):
    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens['input_ids'][0]  # Get the input IDs
    scores = []
    
    for i in range(1, input_ids.size(0) - 1):  # Skip [CLS] and [SEP] tokens
        masked_input_ids = input_ids.clone()
        masked_input_ids[i] = tokenizer.mask_token_id  # Mask the current token

        with torch.no_grad():
            outputs = model(masked_input_ids.unsqueeze(0))  # Add batch dimension

        logits = outputs.logits[0, i]
        probs = torch.softmax(logits, dim=-1)

        original_token_id = input_ids[i]
        score = probs[original_token_id].item()  # Probability of the original word when masked
        
        scores.append(score)

    average_score = sum(scores) / len(scores) * 100  # Convert to percentage
    return average_score, scores

# Compute word-level MLM score for a specific word in a sentence
def compute_word_score(word, sentence, model, tokenizer):
    words = sentence.split()

    if word not in words:
        raise ValueError(f"The word '{word}' is not found in the sentence.")

    index = words.index(word)
    sub_sentence = ' '.join(words[:index + 1])
    tokens = tokenizer(sub_sentence, return_tensors="pt")
    masked_input_ids = tokens['input_ids'].clone()

    word_token_index = tokens['input_ids'][0].size(0) - 2
    masked_input_ids[0, word_token_index] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(masked_input_ids)
    
    logits = outputs.logits
    word_token_id = tokens['input_ids'][0, word_token_index]
    probs = torch.softmax(logits[0, word_token_index], dim=-1)
    score = probs[word_token_id].item()

    return score * 100  # Return as a percentage

# Load existing results from output CSV
def load_existing_results(output_file):
    if not os.path.exists(output_file):
        return set()

    with open(output_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        existing_ngrams = {row['Final_Hybrid_N-Gram'] for row in reader}
    return existing_ngrams

# Get the latest pattern ID from the file
def get_latest_pattern_id(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            pattern_ids = [int(row['Pattern_ID']) for row in reader if row['Pattern_ID'].isdigit()]
            return max(pattern_ids, default=0)
    except FileNotFoundError:
        return 0

# Generate a new pattern ID based on a counter
def generate_pattern_id(counter):
    return f"{counter:06d}"

# Generalize patterns in chunks and save the results incrementally
def generalize_patterns_in_chunks(ngram_list_file, pos_patterns_file, id_array_file, output_file, lexeme_comparison_dict_file, model, tokenizer, chunk_size=2, threshold=80.0):
    print("Loading ngram list...")
    ngram_list = list(load_csv_as_dict(ngram_list_file))  # Load the entire ngram list
    print("Loading POS patterns...")
    pos_patterns = load_csv_as_dict(pos_patterns_file)  # Load the entire POS patterns file
    
    seen_lexeme_comparisons = load_lexeme_comparison_dictionary(lexeme_comparison_dict_file)
    latest_pattern_id_input = get_latest_pattern_id(pos_patterns_file)
    latest_pattern_id_output = get_latest_pattern_id(output_file)
    latest_pattern_id = max(latest_pattern_id_input, latest_pattern_id_output)
    pattern_counter = latest_pattern_id + 1

    existing_hybrid_ngrams = load_existing_results(output_file)
    pos_patterns_dict = {entry['Pattern_ID']: entry['POS_N-Gram'] for entry in pos_patterns}

    for id_array_chunk in load_and_convert_csv_in_chunks(id_array_file, chunk_size):
        pos_comparison_results = []  # Reset the results for each chunk
        print(f"Processing chunk with {len(id_array_chunk)} entries...")
        
        for id_array_entry in id_array_chunk:
            pattern_id = id_array_entry['Pattern_ID']
            rough_pos = pos_patterns_dict.get(pattern_id, None)
            if not rough_pos:
                print(f"No POS pattern found for Pattern_ID {pattern_id}. Skipping...")
                continue

            successful_comparisons = False

            for instance_index, instance_id in enumerate(id_array_entry['ID_Array']):
                instance_id = instance_id.zfill(6)
                instance = next((ngram for ngram in ngram_list if ngram['N-Gram_ID'] == instance_id), None)
                if not instance:
                    continue

                lemma_ngram_sentence = instance.get('Lemma_N-Gram')
                if not lemma_ngram_sentence:
                    continue

                comparison_key = (rough_pos, lemma_ngram_sentence)
                if comparison_key not in seen_lexeme_comparisons:
                    print(f"Computing MLM score for lemma ngram sentence: {lemma_ngram_sentence}...")
                    sequence_mlm_score, _ = compute_mlm_score(lemma_ngram_sentence, model, tokenizer)
                    print(f"Sequence MLM score: {sequence_mlm_score}")

                    if sequence_mlm_score >= threshold:
                        successful_comparisons = True
                        comparison_matrix = ['*'] * len(lemma_ngram_sentence.split())
                        new_pattern = rough_pos.split()
                        words = lemma_ngram_sentence.split()
                        rough_pos_tokens = rough_pos.split()

                        for i, (pos_tag, word) in enumerate(zip(rough_pos_tokens, words)):
                            word_score = compute_word_score(word, lemma_ngram_sentence, model, tokenizer)
                            print(f"Word '{word}' average score: {word_score}")
                            if word_score >= threshold:
                                new_pattern[i] = word
                                comparison_matrix[i] = word

                        final_hybrid_ngram = ' '.join(new_pattern)

                        if final_hybrid_ngram not in existing_hybrid_ngrams:
                            print(f"New hybrid ngram generated: {final_hybrid_ngram}")
                            existing_hybrid_ngrams.add(final_hybrid_ngram)
                            pattern_counter += 1
                            new_pattern_id = generate_pattern_id(pattern_counter)
                            pos_comparison_results.append({
                                'Pattern_ID': new_pattern_id,
                                'POS_N-Gram': rough_pos,
                                'Lexeme_N-Gram': lemma_ngram_sentence,
                                'MLM_Scores': sequence_mlm_score,
                                'Comparison_Replacement_Matrix': ' '.join(comparison_matrix),
                                'Final_Hybrid_N-Gram': final_hybrid_ngram
                            })
                            seen_lexeme_comparisons[comparison_key] = new_pattern_id
                    else:
                        print(f"Sequence MLM score {sequence_mlm_score} did not meet the threshold of {threshold}. Skipping...")
                else:
                    print(f"Comparison already done for rough POS - {rough_pos} and lexeme N-Gram - {lemma_ngram_sentence}")

            if not successful_comparisons:
                print(f"No successful comparisons for pattern ID {pattern_id}. Saving POS pattern only.")
                pos_comparison_results.append({
                    'Pattern_ID': pattern_id,
                    'POS_N-Gram': rough_pos,
                    'Lexeme_N-Gram': '',
                    'MLM_Scores': '',
                    'Comparison_Replacement_Matrix': '',
                    'Final_Hybrid_N-Gram': rough_pos
                })

        # Save the results after processing each chunk
        with open(output_file, 'a', newline='', encoding='utf-8') as file:
            fieldnames = ['Pattern_ID', 'POS_N-Gram', 'Lexeme_N-Gram', 'MLM_Scores', 'Comparison_Replacement_Matrix', 'Final_Hybrid_N-Gram']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:  # If the file is empty, write the header
                writer.writeheader()
            writer.writerows(pos_comparison_results)

    save_lexeme_comparison_dictionary(lexeme_comparison_dict_file, seen_lexeme_comparisons)

# Modify the loop to use chunk processing
for n in range(2, 8):
    ngram_list_file = 'database/ngrams.csv'
    pos_patterns_file = f'database/Generalized/POSTComparison/{n}grams.csv'
    id_array_file = f'database/POS/{n}grams.csv'
    output_file = f'database/Generalized/LexemeComparison/{n}grams.csv'
    comparison_dict_file = 'database/LexComparisonDictionary.txt'

    print(f"Starting generalization for {n}-grams...")
    generalize_patterns_in_chunks(ngram_list_file, pos_patterns_file, id_array_file, output_file, comparison_dict_file, roberta_model, roberta_tokenizer)
    print(f"Finished generalization for {n}-grams.")
