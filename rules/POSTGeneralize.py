import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


# Define the models
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"
roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)
comparison_dict_file = "database/PostComparisonDictionary.txt"

def load_csv(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]
            if not data:
                raise ValueError(f"No data found in {file_path}. Check if file is empty.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return []
    except ValueError as ve:
        print(ve)
        return []
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return []

def load_comparison_dictionary_txt(file_path):
    comparisons = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split("::")
                if len(parts) == 3:
                    rough_pos, detailed_pos_instance, pattern_id = parts
                    comparisons[(rough_pos, detailed_pos_instance)] = pattern_id
                else:
                    print(f"Skipping malformed line: {line.strip()}")
    except FileNotFoundError:
        print(f"Comparison dictionary file not found: {file_path}")
    return comparisons



def save_comparison_dictionary_txt(file_path, dictionary):
    with open(file_path, 'w', encoding='utf-8') as file:
        for key, value in dictionary.items():
            rough_pos, detailed_pos_instance = key  # Assuming key is a tuple
            file.write(f"{rough_pos}::{detailed_pos_instance}::{value}\n")



def compute_mposm_scores(sentence, model, tokenizer):
    mposm_scores = []

    # Tokenize the sentence into words
    words = sentence.split()

    # Iterate over each word index in the sentence
    for index in range(len(words)):
        # Get the sub-sentence up to the indexed word
        sub_sentence = words[:index+1]  # Get words up to the current index
        
        # Mask the current indexed word (using [MASK] token)
        masked_sub_sentence = sub_sentence[:]
        masked_sub_sentence[-1] = tokenizer.mask_token  # Replace the last word with the mask token

        # Join the sub-sentence back into a string
        masked_sentence_str = ' '.join(masked_sub_sentence)

        # Tokenize the masked sub-sentence
        inputs = tokenizer(masked_sentence_str, return_tensors="pt")

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the logits for the masked token position (the last word in the sub-sentence)
        mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        # Get the predicted probabilities for the masked word
        mask_token_logits = outputs.logits[0, mask_token_index, :]

        # Apply softmax to get probabilities
        mask_token_probs = torch.softmax(mask_token_logits, dim=-1)

        # Get the probability score for the actual word (before it was masked)
        actual_word_token_id = tokenizer.convert_tokens_to_ids(words[index])
        actual_word_prob = mask_token_probs[0, actual_word_token_id].item()

        # Append the MPoSM score for this word
        mposm_scores.append(actual_word_prob)

    return mposm_scores


def compare_pos_sequences(rough_pos, detailed_pos, model, tokenizer, threshold=0.80):
    rough_tokens = rough_pos.split()
    detailed_tokens = detailed_pos.split()
    
    if len(rough_tokens) != len(detailed_tokens):
        print(f"Length mismatch: {len(rough_tokens)} (rough) vs {len(detailed_tokens)} (detailed)")
        return None, None, None, None

    rough_scores = []
    detailed_scores = []
    
    # Compute cumulative scores for each token
    rough_scores = compute_mposm_scores(' '.join(rough_tokens), model, tokenizer)
    detailed_scores = compute_mposm_scores(' '.join(detailed_tokens), model, tokenizer)


    comparison_matrix = []
    for i in range(len(detailed_tokens)):
        rough_token = rough_tokens[i]
        detailed_token = detailed_tokens[i]
        rough_score = rough_scores[i]
        detailed_score = detailed_scores[i]

        if rough_score != 0 and detailed_score / rough_score >= threshold:
            comparison_matrix.append(detailed_token)
        else:
            comparison_matrix.append('*')

    new_pattern = [comparison_matrix[i] if comparison_matrix[i] != '*' else rough_tokens[i] for i in range(len(rough_tokens))]
    comparison_matrix_str = ' '.join(comparison_matrix)
    new_pattern_str = ' '.join(new_pattern)

    return rough_scores, detailed_scores, comparison_matrix_str, new_pattern_str


def generate_pattern_id(counter):
    return f"{counter:06d}"

def collect_existing_patterns(file_path):
    patterns = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['POS_N-Gram']:
                    patterns.add(row['POS_N-Gram'])
    except FileNotFoundError:
        pass
    return patterns

def get_latest_pattern_id(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            pattern_ids = [int(row['Pattern_ID']) for row in reader if row['Pattern_ID'].isdigit()]
            return max(pattern_ids, default=0)
    except FileNotFoundError:
        return 0

def process_pos_patterns(pos_patterns_file, generated_ngrams_file, pattern_file, output_file, model, tokenizer, threshold=0.80):
    pos_patterns = load_csv(pos_patterns_file)
    generated_ngrams = load_csv(generated_ngrams_file)

    print(f"Loaded {len(pos_patterns)} POS patterns")
    print(f"Loaded {len(generated_ngrams)} generated n-grams")

    existing_patterns_input = set()
    existing_patterns_output = collect_existing_patterns(output_file)

    # Collect existing patterns from input CSV
    for pattern in pos_patterns:
        if pattern['RoughPOS_N-Gram']:
            existing_patterns_input.add(pattern['RoughPOS_N-Gram'])
        if pattern['DetailedPOS_N-Gram']:
            existing_patterns_input.add(pattern['DetailedPOS_N-Gram'])

    latest_pattern_id_input = get_latest_pattern_id(pos_patterns_file)
    latest_pattern_id_output = get_latest_pattern_id(output_file)
    latest_pattern_id = max(latest_pattern_id_input, latest_pattern_id_output)
    pattern_counter = latest_pattern_id + 1

    pos_comparison_results = []
    new_patterns = []
    seen_patterns = {}

    # Load comparison dictionary
    seen_comparisons = load_comparison_dictionary_txt(comparison_dict_file)


    for pattern in pos_patterns:
        pattern_id = pattern['Pattern_ID']
        rough_pos = pattern['RoughPOS_N-Gram']
        detailed_pos = pattern['DetailedPOS_N-Gram']
        id_array = pattern['ID_Array'].split(',') if pattern['ID_Array'] else []

        if rough_pos and rough_pos in existing_patterns_input:
            instances = [ngram for ngram in generated_ngrams if ngram['N-Gram_ID'] in id_array]
            for instance in instances:
                detailed_pos_instance = instance['DetailedPOS_N-Gram']
                comparison_key = f"{rough_pos}::{detailed_pos_instance}"
                
                # Skip if this comparison was already done
                if detailed_pos_instance and comparison_key not in seen_comparisons:
                    rough_scores, detailed_scores, comparison_matrix, new_pattern = compare_pos_sequences(rough_pos, detailed_pos_instance, model, tokenizer, threshold)
                    
                    if rough_scores and detailed_scores:
                        if new_pattern not in existing_patterns_input and new_pattern not in existing_patterns_output:
                            new_pattern_id = generate_pattern_id(pattern_counter)
                            seen_patterns[new_pattern] = new_pattern_id
                            seen_comparisons[comparison_key] = new_pattern_id  # Log this comparison
                            pattern_counter += 1
                            pos_comparison_results.append({
                                'Pattern_ID': new_pattern_id,
                                'RoughPOS_N-Gram': rough_pos,
                                'RPOSN_Freq': None,
                                'DetailedPOS_N-Gram': detailed_pos_instance,
                                'DPOSN_Freq': None,
                                'Comparison_Replacement_Matrix': comparison_matrix,
                                'POS_N-Gram': new_pattern
                            })
                            new_patterns.append({
                                'Pattern_ID': new_pattern_id,
                                'RoughPOS_N-Gram': rough_pos,
                                'DetailedPOS_N-Gram': detailed_pos_instance,
                                'Frequency': len(id_array),
                                'ID_Array': ','.join(id_array)
                            })
                            print(f"Comparison made: Rough POS - {rough_pos}, Detailed POS - {detailed_pos_instance}")
                    else:
                        seen_comparisons[comparison_key] = None  # Mark as failed comparison
                else:
                    print(f"Comparison already done for Rough POS - {rough_pos} and Detailed POS - {detailed_pos_instance}")

            if rough_pos not in seen_patterns and rough_pos not in existing_patterns_output:
                pos_comparison_results.append({
                    'Pattern_ID': pattern_id,
                    'RoughPOS_N-Gram': rough_pos,
                    'RPOSN_Freq': None,
                    'DetailedPOS_N-Gram': None,
                    'DPOSN_Freq': None,
                    'Comparison_Replacement_Matrix': None,
                    'POS_N-Gram': rough_pos
                })
                seen_patterns[rough_pos] = pattern_id
        
        elif detailed_pos and detailed_pos in existing_patterns_input:
            overlap_patterns = [p for p in pos_patterns if p['Pattern_ID'] != pattern_id and any(id in id_array for id in p['ID_Array'].split(','))]
            for overlap_pattern in overlap_patterns:
                overlap_rough_pos = overlap_pattern['RoughPOS_N-Gram']
                comparison_key = (overlap_rough_pos, detailed_pos)
                
                # Skip if this comparison was already done
                if overlap_rough_pos and comparison_key not in seen_comparisons:
                    rough_scores, detailed_scores, comparison_matrix, new_pattern = compare_pos_sequences(overlap_rough_pos, detailed_pos, model, tokenizer, threshold)
                    
                    if rough_scores and detailed_scores:
                        if new_pattern not in existing_patterns_input and new_pattern not in existing_patterns_output:
                            new_pattern_id = generate_pattern_id(pattern_counter)
                            seen_patterns[new_pattern] = new_pattern_id
                            seen_comparisons[comparison_key] = new_pattern_id  # Log this comparison
                            pattern_counter += 1
                            pos_comparison_results.append({
                                'Pattern_ID': new_pattern_id,
                                'RoughPOS_N-Gram': overlap_rough_pos,
                                'RPOSN_Freq': None,
                                'DetailedPOS_N-Gram': detailed_pos,
                                'DPOSN_Freq': None,
                                'Comparison_Replacement_Matrix': comparison_matrix,
                                'POS_N-Gram': new_pattern
                            })
                            new_patterns.append({
                                'Pattern_ID': new_pattern_id,
                                'RoughPOS_N-Gram': overlap_rough_pos,
                                'DetailedPOS_N-Gram': detailed_pos,
                                'Frequency': len(id_array),
                                'ID_Array': ','.join(id_array)
                            })
                            print(f"Comparison made: Rough POS - {overlap_rough_pos}, Detailed POS - {detailed_pos}")
                    else:
                        seen_comparisons[comparison_key] = None  # Mark as failed comparison
                else:
                    print(f"Comparison already done for Rough POS - {overlap_rough_pos} and Detailed POS - {detailed_pos}")

            if detailed_pos not in seen_patterns and detailed_pos not in existing_patterns_output:
                pos_comparison_results.append({
                    'Pattern_ID': pattern_id,
                    'RoughPOS_N-Gram': None,
                    'RPOSN_Freq': None,
                    'DetailedPOS_N-Gram': detailed_pos,
                    'DPOSN_Freq': None,
                    'Comparison_Replacement_Matrix': None,
                    'POS_N-Gram': detailed_pos
                })
                seen_patterns[detailed_pos] = pattern_id

    # Append new results and patterns to the output

    # Save the updated results and patterns
    save_comparison_dictionary_txt(comparison_dict_file, seen_comparisons)
    existing_data = []
    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            existing_data = [row for row in reader]
    except FileNotFoundError:
        pass  # No existing data

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'RPOSN_Freq', 'DetailedPOS_N-Gram', 'DPOSN_Freq', 'Comparison_Replacement_Matrix', 'POS_N-Gram']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in existing_data + pos_comparison_results:
            writer.writerow(result)

    with open(pattern_file, 'a', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'Frequency', 'ID_Array']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for new_pattern in new_patterns:
            writer.writerow(new_pattern)

for n in range(6, 8):
    ngram_csv = 'database/ngrams.csv'
    pattern_csv = f'database/POS/{n}grams.csv'
    output_csv = f'database/Generalized/POSTComparison/{n}grams.csv'

    print(f"Processing n-gram size: {n}")  # Debug statement

    process_pos_patterns(pattern_csv, ngram_csv, pattern_csv, output_csv, roberta_model, roberta_tokenizer)
