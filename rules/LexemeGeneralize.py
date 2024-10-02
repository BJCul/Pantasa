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

def load_csv(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    print(f"Data loaded from {file_path}. Number of rows: {len(data)}")
    return data

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

def save_lexeme_comparison_dictionary(file_path, dictionary):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for key, value in dictionary.items():
                rough_pos, lexeme_ngram = key
                file.write(f"{rough_pos}::{lexeme_ngram}::{value}\n")
    except Exception as e:
        print(f"Error writing lexeme comparison dictionary: {e}")

def convert_id_array(id_array_str):
    if id_array_str is None:
        return []  # Return an empty list or handle it accordingly
    return id_array_str.strip("[]'").replace("'", "").split(', ')

def load_and_convert_csv(file_path):
    data = load_csv(file_path)
    for entry in data:
        print(f"Processing entry: {entry}")
        entry['ID_Array'] = convert_id_array(entry.get('ID_Array', ''))
    return data

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

def compute_word_score(word, sentence, model, tokenizer):
    # Split the sentence into words
    words = sentence.split()

    # Check if the word is in the sentence
    if word not in words:
        raise ValueError(f"The word '{word}' is not found in the sentence.")

    # Find the index of the word in the sentence
    index = words.index(word)

    # Create a sub-sentence up to the current word
    sub_sentence = ' '.join(words[:index + 1])
    
    # Tokenize the sub-sentence and mask the word at the current index
    tokens = tokenizer(sub_sentence, return_tensors="pt")
    masked_input_ids = tokens['input_ids'].clone()

    # Find the token ID corresponding to the word at the current index
    word_token_index = tokens['input_ids'][0].size(0) - 2  # Get second-to-last token (ignores [SEP] and [CLS])
    masked_input_ids[0, word_token_index] = tokenizer.mask_token_id  # Mask the indexed word

    # Get model output for masked sub-sentence
    with torch.no_grad():
        outputs = model(masked_input_ids)
    
    # Extract the logits for the masked word and calculate its probability
    logits = outputs.logits
    word_token_id = tokens['input_ids'][0, word_token_index]  # The original token ID of the indexed word
    probs = torch.softmax(logits[0, word_token_index], dim=-1)
    score = probs[word_token_id].item()  # Probability of the original word when masked

    return score * 100  # Return as a percentage

def load_existing_results(output_file):
    if not os.path.exists(output_file):
        return set()

    with open(output_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        existing_ngrams = {row['Final_Hybrid_N-Gram'] for row in reader}
    return existing_ngrams

def get_latest_pattern_id(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            pattern_ids = [int(row['Pattern_ID']) for row in reader if row['Pattern_ID'].isdigit()]
            return max(pattern_ids, default=0)
    except FileNotFoundError:
        return 0

def generate_pattern_id(counter):
    return f"{counter:06d}"

def generalize_patterns(ngram_list_file, pos_patterns_file, id_array_file, output_file, lexeme_comparison_dict_file, model, tokenizer, threshold=80.0):
    print("Loading ngram list...")
    ngram_list = load_csv(ngram_list_file)
    print("Loading POS patterns...")
    pos_patterns = load_csv(pos_patterns_file)
    print("Loading ID array data...")
    id_array_data = load_and_convert_csv(id_array_file)
    
    seen_lexeme_comparisons = load_lexeme_comparison_dictionary(lexeme_comparison_dict_file)

    latest_pattern_id_input = get_latest_pattern_id(pos_patterns_file)
    latest_pattern_id_output = get_latest_pattern_id(output_file)
    latest_pattern_id = max(latest_pattern_id_input, latest_pattern_id_output)
    pattern_counter = latest_pattern_id + 1

    existing_hybrid_ngrams = load_existing_results(output_file)

    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            pos_comparison_results = [row for row in reader]
    except FileNotFoundError:
        pos_comparison_results = []  # If file doesn't exist, initialize with an empty list

    pos_patterns_dict = {entry['Pattern_ID']: entry['POS_N-Gram'] for entry in pos_patterns}

    for id_array_index, id_array_entry in enumerate(id_array_data):
        print(f"Processing ID array entry {id_array_index + 1}/{len(id_array_data)}...")
        pattern_id = id_array_entry['Pattern_ID']
        rough_pos = pos_patterns_dict.get(pattern_id, None)
        if not rough_pos:
            print(f"No POS pattern found for Pattern_ID {pattern_id}. Skipping...")
            continue

        # Initialize a flag to track if we have successful comparisons
        successful_comparisons = False

        for instance_index, instance_id in enumerate(id_array_entry['ID_Array']):
            instance_id = instance_id.zfill(6)
            print(f"Processing instance {instance_index + 1}/{len(id_array_entry['ID_Array'])} for pattern ID {pattern_id}...")

            instance = next((ngram for ngram in ngram_list if ngram['N-Gram_ID'] == instance_id), None)
            if not instance:
                print(f"No instance found for ID {instance_id}.")
                continue

            lemma_ngram_sentence = instance.get('Lemma_N-Gram')  # Use `.get()` to avoid KeyError
            if not lemma_ngram_sentence:
                print(f"No lemma ngram sentence found for instance ID {instance_id}. Skipping...")
                continue

            comparison_key = (rough_pos, lemma_ngram_sentence)
            if comparison_key not in seen_lexeme_comparisons:
                print(f"Computing MLM score for lemma ngram sentence: {lemma_ngram_sentence}...")
                initial_mlm_score = compute_mlm_score(lemma_ngram_sentence, model, tokenizer)
                sequence_mlm_score = initial_mlm_score[0]
                print(f"Sequence MLM score: {sequence_mlm_score}")

                if sequence_mlm_score >= threshold:
                    successful_comparisons = True  # Flagging that a successful comparison was made
                    print(f"Sequence MLM score {sequence_mlm_score} meets the threshold {threshold}. Computing individual word scores...")
                    comparison_matrix = ['*'] * len(lemma_ngram_sentence.split())
                    new_pattern = rough_pos.split()
                    words = lemma_ngram_sentence.split()
                    rough_pos_tokens = rough_pos.split()

                    if len(words) != len(rough_pos_tokens):
                        print(f"Length mismatch between words and POS tokens for instance ID {instance_id}. Skipping...")
                        continue

                    for i, (pos_tag, word) in enumerate(zip(rough_pos_tokens, words)):
                        word_score = compute_word_score(word, lemma_ngram_sentence, model, tokenizer)
                        print(f"Word '{word}' average score: {word_score}")
                        if word_score >= threshold:
                            new_pattern[i] = word
                            comparison_matrix[i] = word
                        else:
                            new_pattern[i] = pos_tag  # Restore the original POS tag if the word does not meet the score

                    final_hybrid_ngram = ' '.join(new_pattern)

                    if final_hybrid_ngram not in existing_hybrid_ngrams:
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
                        seen_lexeme_comparisons[comparison_key] = new_pattern_id  # Update the comparison dictionary
                    else:
                        print(f"Hybrid ngram '{final_hybrid_ngram}' already exists. Skipping...")
                else:
                    print(f"Sequence MLM score {sequence_mlm_score} does not meet the threshold {threshold}. Skipping...")
            else:
                print(f"Comparison already done for rough POS - {rough_pos} and lexeme N-Gram - {lemma_ngram_sentence}")

        # If no successful comparison was made, still append the POS pattern without any final hybrid n-gram
        if not successful_comparisons:
            pos_comparison_results.append({
                'Pattern_ID': pattern_id,
                'POS_N-Gram': rough_pos,
                'Lexeme_N-Gram': '',  # No lexeme comparison succeeded
                'MLM_Scores': '',
                'Comparison_Replacement_Matrix': '',
                'Final_Hybrid_N-Gram': rough_pos  # Keep the original POS pattern
            })

    save_lexeme_comparison_dictionary(lexeme_comparison_dict_file, seen_lexeme_comparisons)

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'POS_N-Gram', 'Lexeme_N-Gram', 'MLM_Scores', 'Comparison_Replacement_Matrix', 'Final_Hybrid_N-Gram']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pos_comparison_results)

    print(f"Results saved to {output_file}")

for n in range(2, 8):
    ngram_list_file = 'database/ngrams.csv'
    pos_patterns_file = f'database/Generalized/POSTComparison/{n}grams.csv'
    id_array_file = f'database/POS/{n}grams.csv'
    output_file = f'database/Generalized/LexemeComparison/{n}grams.csv'
    comparison_dict_file = 'database/LexComparisonDictionary.txt'

    print(f"Starting generalization for {n}-grams...")
    generalize_patterns(ngram_list_file, pos_patterns_file, id_array_file, output_file, comparison_dict_file, roberta_model, roberta_tokenizer)
    print(f"Finished generalization for {n}-grams.")
