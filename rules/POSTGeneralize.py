import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
from hngram_counter import instance_collector, get_ngram_size_from_pattern_id  # Import functions from hngram_counter.py

# Define the models
tagalog_roberta_model = "model/final_model"
tagalog_roberta_tokenizer = "model/final_tokenizer"
roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_tokenizer)
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)
comparison_dict_file = "rules/database/PostComparisonDictionary.txt"

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
    words = sentence.split()

    for index in range(len(words)):
        sub_sentence = words[:index + 1]
        masked_sub_sentence = sub_sentence[:]
        masked_sub_sentence[-1] = tokenizer.mask_token
        masked_sentence_str = ' '.join(masked_sub_sentence)
        inputs = tokenizer(masked_sentence_str, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
        mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        mask_token_logits = outputs.logits[0, mask_token_index, :]
        mask_token_probs = torch.softmax(mask_token_logits, dim=-1)

        actual_word_token_id = tokenizer.convert_tokens_to_ids(words[index])
        actual_word_prob = mask_token_probs[0, actual_word_token_id].item()
        mposm_scores.append(actual_word_prob)

    return mposm_scores

def compare_pos_sequences(rough_pos, detailed_pos, model, tokenizer, threshold=0.80):
    rough_tokens = rough_pos.split()
    detailed_tokens = detailed_pos.split()
    
    if len(rough_tokens) != len(detailed_tokens):
        print(f"Length mismatch: {len(rough_tokens)} (rough) vs {len(detailed_tokens)} (detailed)")
        return None, None, None, None

    rough_scores = compute_mposm_scores(' '.join(rough_tokens), model, tokenizer)
    detailed_scores = compute_mposm_scores(' '.join(detailed_tokens), model, tokenizer)

    comparison_matrix = []
    for i in range(len(detailed_tokens)):
        rough_score = rough_scores[i]
        detailed_score = detailed_scores[i]
        comparison_matrix.append(detailed_tokens[i] if rough_score != 0 and detailed_score / rough_score >= threshold else '*')

    new_pattern = [comparison_matrix[i] if comparison_matrix[i] != '*' else rough_tokens[i] for i in range(len(rough_tokens))]
    return rough_scores, detailed_scores, ' '.join(comparison_matrix), ' '.join(new_pattern)

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

def process_pos_patterns(pos_patterns_file, generated_ngrams_file, pattern_file, output_file, model, tokenizer, threshold=0.50):
    print(f"Loading POS patterns from: {pos_patterns_file}")
    pos_patterns = load_csv(pos_patterns_file)
    print(f"Loaded POS patterns: {pos_patterns}")
    generated_ngrams = pd.read_csv(generated_ngrams_file)

    existing_patterns_output = collect_existing_patterns(output_file)
    latest_pattern_id = max(get_latest_pattern_id(pos_patterns_file), get_latest_pattern_id(output_file))
    pattern_counter = latest_pattern_id + 1

    pos_comparison_results = []
    new_patterns = []
    seen_comparisons = load_comparison_dictionary_txt(comparison_dict_file)
    print(f"Loaded seen_comparisons: {seen_comparisons}")
    

    for pattern in pos_patterns:
        pattern_id = pattern['Pattern_ID']
        rough_pos = pattern['RoughPOS_N-Gram']
        detailed_pos = pattern['DetailedPOS_N-Gram']
        print(f"RoughPOS: {rough_pos}, DetailedPOS: {detailed_pos}")
        id_array = pattern['ID_Array'].split(',') if pattern['ID_Array'] else []

        if rough_pos:
            rough_pattern_size = get_ngram_size_from_pattern_id(pattern_id)
            rough_pos_matches = instance_collector(rough_pos, generated_ngrams, rough_pattern_size)
            print(f"Matches for RoughPOS: {rough_pos} -> {rough_pos_matches}")
            rough_pos_frequency = rough_pos_matches.shape[0]
        else:
            rough_pos_frequency = None

        if detailed_pos:
            detailed_pattern_size = get_ngram_size_from_pattern_id(pattern_id)
            detailed_pos_matches = instance_collector(detailed_pos, generated_ngrams, detailed_pattern_size)
            detailed_pos_frequency = detailed_pos_matches.shape[0]
        else:
            detailed_pos_frequency = None

        if rough_pos and detailed_pos:
            comparison_key = f"{rough_pos}::{detailed_pos}"
            print(f"RoughPOS: {rough_pos}, DetailedPOS: {detailed_pos}")
            if comparison_key not in seen_comparisons:
                print(f"Processing new comparison: {comparison_key}")
                rough_scores, detailed_scores, comparison_matrix, new_pattern = compare_pos_sequences(rough_pos, detailed_pos, model, tokenizer, threshold)
                if new_pattern not in existing_patterns_output:
                    new_pattern_id = generate_pattern_id(pattern_counter)
                    seen_comparisons[comparison_key] = new_pattern_id
                    existing_patterns_output.add(new_pattern)
                    pattern_counter += 1
                    print(f"New pattern being added: {new_pattern}, ID: {new_pattern_id}")
                    pos_comparison_results.append({
                        'Pattern_ID': new_pattern_id,
                        'RoughPOS_N-Gram': rough_pos,
                        'RPOSN_Freq': rough_pos_frequency,
                        'DetailedPOS_N-Gram': detailed_pos,
                        'DPOSN_Freq': detailed_pos_frequency,
                        'Comparison_Replacement_Matrix': comparison_matrix,
                        'POS_N-Gram': new_pattern
                    })
                    new_patterns.append({
                        'Pattern_ID': new_pattern_id,
                        'RoughPOS_N-Gram': rough_pos,
                        'DetailedPOS_N-Gram': detailed_pos,
                        'Frequency': len(id_array),
                        'ID_Array': ','.join(id_array)
                    })
                    print(f"Comparison made: Rough POS - {rough_pos}, Detailed POS - {detailed_pos}")
            else:
                print(f"Comparison already done for Rough POS - {rough_pos} and Detailed POS - {detailed_pos}")

        else:
            pos_comparison_results.append({
                'Pattern_ID': pattern_id,
                'RoughPOS_N-Gram': rough_pos or None,
                'RPOSN_Freq': rough_pos_frequency,
                'DetailedPOS_N-Gram': detailed_pos or None,
                'DPOSN_Freq': detailed_pos_frequency,
                'Comparison_Replacement_Matrix': None,
                'POS_N-Gram': rough_pos or detailed_pos
            })

    save_comparison_dictionary_txt(comparison_dict_file, seen_comparisons)

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'RPOSN_Freq', 'DetailedPOS_N-Gram', 'DPOSN_Freq', 'Comparison_Replacement_Matrix', 'POS_N-Gram']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pos_comparison_results)

    with open(pattern_file, 'a', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'Frequency', 'ID_Array']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(new_patterns)

for n in range(4, 5):
    ngram_csv = 'rules/database/ngram.csv'
    pattern_csv = f'rules/database/POS/{n}grams.csv'
    output_csv = f'rules/database/Generalized/POSTComparison/{n}grams.csv'

    print(f"Processing n-gram size: {n}")
    process_pos_patterns(pattern_csv, ngram_csv, pattern_csv, output_csv, roberta_model, roberta_tokenizer)