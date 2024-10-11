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
batch_size=2

def load_csv_in_batches(file_path, batch_size):
    """
    Load CSV file in batches to handle large datasets.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        batch = []
        for index, row in enumerate(reader):
            batch.append(row)
            if (index + 1) % batch_size == 0:
                yield batch
                batch = []
        if batch:
            yield batch

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

def subword_penalty_score(score, word, tokenizer):
    tokenized_word = tokenizer(word, return_tensors="pt")['input_ids'][0]
    num_tokens = len(tokenized_word)
    penalized_score = score / num_tokens
    return penalized_score

def compute_mlm_score(sentence, model, tokenizer):
    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    scores = []
    
    for i in range(1, input_ids.size(0) - 1):
        masked_input_ids = input_ids.clone()
        masked_input_ids[i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input_ids.unsqueeze(0))

        logits = outputs.logits[0, i]
        probs = torch.softmax(logits, dim=-1)

        original_token_id = input_ids[i]
        score = probs[original_token_id].item()

        word = tokenizer.decode([original_token_id]).strip()
        penalized_score = subword_penalty_score(score, word, tokenizer)
        
        scores.append(penalized_score)

    average_score = sum(scores) / len(scores) * 100
    return average_score, scores

def generalize_patterns_batch(ngram_list_file, pos_patterns_file, id_array_file, output_file, lexeme_comparison_dict_file, model, tokenizer, batch_size=batch_size, threshold=80.0):
    print("Loading ngram list...")
    ngram_list = load_csv_in_batches(ngram_list_file, batch_size)
    print("Loading POS patterns...")
    pos_patterns = load_csv_in_batches(pos_patterns_file, batch_size)
    print("Loading ID array data...")
    id_array_batches = load_csv_in_batches(id_array_file, batch_size)
    
    seen_lexeme_comparisons = load_lexeme_comparison_dictionary(lexeme_comparison_dict_file)
    pos_comparison_results = []

    for id_array_batch in id_array_batches:
        for id_array_entry in id_array_batch:
            pattern_id = id_array_entry['Pattern_ID']
            for instance_id in convert_id_array(id_array_entry.get('ID_Array', '')):
                instance = next((ngram for ngram in ngram_list if ngram['N-Gram_ID'] == instance_id.zfill(6)), None)
                if not instance:
                    continue

                lemma_ngram_sentence = instance.get('Lemma_N-Gram', '')
                if (pattern_id, lemma_ngram_sentence) not in seen_lexeme_comparisons:
                    score, _ = compute_mlm_score(lemma_ngram_sentence, model, tokenizer)
                    if score >= threshold:
                        pos_comparison_results.append({
                            'Pattern_ID': pattern_id,
                            'POS_N-Gram': instance.get('POS_N-Gram', ''),
                            'Lexeme_N-Gram': lemma_ngram_sentence,
                            'MLM_Scores': score,
                            'Comparison_Replacement_Matrix': '',  # Implement as needed
                            'Final_Hybrid_N-Gram': lemma_ngram_sentence  # Simplified
                        })
                        seen_lexeme_comparisons[(pattern_id, lemma_ngram_sentence)] = pattern_id

        # Save results and dictionary after processing each batch
        with open(output_file, 'a', newline='', encoding='utf-8') as file:
            fieldnames = ['Pattern_ID', 'POS_N-Gram', 'Lexeme_N-Gram', 'MLM_Scores', 'Comparison_Replacement_Matrix', 'Final_Hybrid_N-Gram']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if os.stat(output_file).st_size == 0:
                writer.writeheader()
            writer.writerows(pos_comparison_results)

        save_lexeme_comparison_dictionary(lexeme_comparison_dict_file, seen_lexeme_comparisons)
        pos_comparison_results = []  # Reset after saving

for n in range(2, 8):
    ngram_list_file = 'database/ngrams.csv'
    pos_patterns_file = f'database/Generalized/POSTComparison/{n}grams.csv'
    id_array_file = f'database/POS/{n}grams.csv'
    output_file = f'database/Generalized/LexemeComparison/{n}grams.csv'
    comparison_dict_file = 'database/LexComparisonDictionary.txt'

    print(f"Starting generalization for {n}-grams...")
    generalize_patterns_batch(ngram_list_file, pos_patterns_file, id_array_file, output_file, comparison_dict_file, roberta_model, roberta_tokenizer)
    print(f"Finished generalization for {n}-grams.")
