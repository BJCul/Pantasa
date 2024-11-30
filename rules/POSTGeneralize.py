import csv
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
import ast
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
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

def generate_pattern_id(size,counter):
    return f"{size}{counter:05d}"

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
            pattern_ids = [
                int(row['Pattern_ID']) 
                for row in reader 
                if row['Pattern_ID'].isdigit() and len(row['Pattern_ID']) == 6 and row['Pattern_ID'].startswith('5')
            ]
            return max(pattern_ids, default=500000)  # Starting from a default base for safety
    except FileNotFoundError:
        return 500000  # Default starting point if file not found

def process_pos_patterns_chunk(size, pos_patterns_chunk, generated_ngrams_file, output_file, pattern_file, model, tokenizer, seen_comparisons, pattern_counter, threshold=0.50):
    generated_ngrams = pd.read_csv(generated_ngrams_file)
    existing_patterns_output = collect_existing_patterns(output_file)
    
    # Replace NaN values in the chunk with empty strings
    pos_patterns_chunk = pos_patterns_chunk.fillna('')

    pos_comparison_results = []
    new_patterns = []

    for _, pattern in pos_patterns_chunk.iterrows():
        pattern_id = pattern['Pattern_ID']
        rough_pos = pattern['RoughPOS_N-Gram']
        detailed_pos = pattern['DetailedPOS_N-Gram']
        id_array_str = pattern['ID_Array']

        # Safely parse the ID_Array string into a list using ast.literal_eval
        if id_array_str:
            try:
                id_array = ast.literal_eval(id_array_str)
                if not isinstance(id_array, list):
                    id_array = []  # Ensure id_array is a list even if parsing goes wrong
            except (ValueError, SyntaxError):
                id_array = []
        else:
            id_array = []

        # Direct lookup using ID_Array for RoughPOS
        if rough_pos and id_array:
            rough_pattern_size = get_ngram_size_from_pattern_id(pattern_id)
            # Convert id_array from string to integers if applicable
            id_array = [int(id_.strip()) for id_ in id_array]
            rough_pos_matches = generated_ngrams[generated_ngrams['N-Gram_ID'].isin(id_array)]
            rough_pos_frequency = rough_pos_matches.shape[0]
            print(f"Matches for RoughPOS: {rough_pos} -> {rough_pos_matches}")
            pos_comparison_results.append({
                'Pattern_ID': pattern_id,
                'RoughPOS_N-Gram': rough_pos or None,
                'RPOSN_Freq': rough_pos_frequency,
                'DetailedPOS_N-Gram': None,
                'DPOSN_Freq': None,
                'Comparison_Replacement_Matrix': None,
                'POS_N-Gram': rough_pos
            })

            for _, rough_pos_match in rough_pos_matches.iterrows():
                match_pos =  rough_pos_match['DetailedPOS_N-Gram']               
                comparison_key = f"{rough_pos}::{match_pos}"
                print(f"RoughPOS: {rough_pos}, DetailedPOS: {match_pos}")
                if comparison_key not in seen_comparisons:
                    print(f"Processing new comparison: {comparison_key}")
                    _, _, comparison_matrix, new_pattern = compare_pos_sequences(rough_pos, match_pos, model, tokenizer, threshold)
                    if new_pattern not in existing_patterns_output:
                        new_pattern_id = generate_pattern_id(size, pattern_counter)
                        seen_comparisons[comparison_key] = new_pattern_id
                        existing_patterns_output.add(new_pattern)
                        pattern_counter += 1

                        new_pattern_matches = instance_collector (new_pattern,  generated_ngrams, rough_pattern_size)
                        ngram_id_list = new_pattern_matches['N-Gram_ID'].tolist()


                        print(f"New pattern being added: {new_pattern}, ID: {new_pattern_id}")
                        pos_comparison_results.append({
                            'Pattern_ID': new_pattern_id,
                            'RoughPOS_N-Gram': rough_pos,
                            'RPOSN_Freq': rough_pos_frequency,
                            'DetailedPOS_N-Gram': match_pos,
                            'DPOSN_Freq': len(ngram_id_list),
                            'Comparison_Replacement_Matrix': comparison_matrix,
                            'POS_N-Gram': new_pattern
                        })

                        new_patterns.append({
                            'Pattern_ID': new_pattern_id,
                            'RoughPOS_N-Gram': rough_pos,
                            'DetailedPOS_N-Gram': match_pos,
                            'Frequency': len(ngram_id_list),
                            'ID_Array': ','.join(map(str, ngram_id_list))
                        })
                        print(f"Comparison made: Rough POS - {rough_pos}, Detailed POS - {detailed_pos}")
                else:
                    print(f"Comparison already done for Rough POS - {rough_pos} and Detailed POS - {detailed_pos}")

        else:
            # Convert id_array from string to integers if applicable
            id_array = [int(id_.strip()) for id_ in id_array]
            detailed_pos_matches = generated_ngrams[generated_ngrams['N-Gram_ID'].isin(id_array)]
            detailed_pos_frequency = detailed_pos_matches.shape[0]
            pos_comparison_results.append({
                'Pattern_ID': pattern_id,
                'RoughPOS_N-Gram': None,
                'RPOSN_Freq': None,
                'DetailedPOS_N-Gram': detailed_pos,
                'DPOSN_Freq': detailed_pos_frequency,
                'Comparison_Replacement_Matrix': None,
                'POS_N-Gram': detailed_pos
            })

    # Write comparison results to output file
    if pos_comparison_results:
        with open(output_file, 'a', newline='', encoding='utf-8') as file:
            fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'RPOSN_Freq', 'DetailedPOS_N-Gram', 'DPOSN_Freq', 'Comparison_Replacement_Matrix', 'POS_N-Gram']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if os.stat(output_file).st_size == 0:
                writer.writeheader()  # Write header if the file is empty
            writer.writerows(pos_comparison_results)

    # Write new patterns to pattern file
    if new_patterns:
        with open(pattern_file, 'a', newline='', encoding='utf-8') as file:
            fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'Frequency', 'ID_Array']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if os.stat(pattern_file).st_size == 0:
                writer.writeheader()  # Write header if the file is empty
            writer.writerows(new_patterns)

    return pattern_counter


if __name__ == "__main__":
    ngram_csv = 'rules/database/ngram.csv'
    for n in range(5, 6):
        pattern_csv = f'rules/database/POS/{n}grams.csv'
        output_csv = f'rules/database/Generalized/POSTComparison/{n}grams.csv'
        chunk_size = 5000

        # Prepare the output files with headers if they are empty
        if not os.path.exists(output_csv):
            with open(output_csv, 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'RPOSN_Freq', 'DetailedPOS_N-Gram', 'DPOSN_Freq', 'Comparison_Replacement_Matrix', 'POS_N-Gram']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
        
        if not os.path.exists(pattern_csv):
            with open(pattern_csv, 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'Frequency', 'ID_Array']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

        # Load the existing comparison dictionary and latest pattern ID
        seen_comparisons = load_comparison_dictionary_txt(comparison_dict_file)
        latest_pattern_id = get_latest_pattern_id(pattern_csv)
        pattern_counter = latest_pattern_id + 1

        # Process CSV in chunks in parallel
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for chunk in pd.read_csv(pattern_csv, chunksize=chunk_size):
                futures.append(executor.submit(
                    process_pos_patterns_chunk,
                    n,
                    chunk,
                    ngram_csv,
                    output_csv,
                    pattern_csv,
                    roberta_model,
                    roberta_tokenizer,
                    seen_comparisons,
                    pattern_counter
                ))

            for future in futures:
                pattern_counter = future.result()

        # Save updated comparison dictionary
        save_comparison_dictionary_txt(comparison_dict_file, seen_comparisons)