import csv
csv.field_size_limit(10**7)  
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.nn.functional import cosine_similarity
import os
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model
global_max_score = 0
global_min_score = 0
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"
print("Loading tokenizer...")
roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)
print("Loading model...")
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model).to(device)
print("Model and tokenizer loaded successfully.")

def load_csv(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    print(f"Data loaded from {file_path}. Number of rows: {len(data)}")
    return data

def load_frequency_dict(file_path):
    frequency_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                word = row.get('Word')  # Adjust based on column name
                frequency = int(row.get('Frequency', 0))  # Adjust column name as well
                if word:
                    frequency_dict[word] = frequency
    except FileNotFoundError:
        print(f"Frequency dictionary file not found: {file_path}")
    return frequency_dict

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
        
def redo_escape_and_wrap(sentence):
    """Escape double quotes for CSV compatibility and wrap with quotes if necessary."""
    sentence = sentence.replace('"', '""')
    if ',' in sentence or (sentence.startswith('""') and sentence.endswith('""')):
        sentence = f'"{sentence}"'
    return sentence

def undo_escape_and_wrap(sentence):
    """Revert double quotes and wrapping for processing."""
    if sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence[1:-1]
    return sentence.replace('""', '"')


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

def get_subword_embeddings(word, model, tokenizer):
    tokens = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    subword_embeddings = outputs.hidden_states[-1][0][1:-1]  # Ignore CLS and SEP tokens
    return subword_embeddings

def compute_complexity_score(word, model, tokenizer, frequency_dict):
    # Get the whole word embedding
    tokens = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    whole_word_embedding = torch.mean(outputs.hidden_states[-1][0][1:-1], dim=0)

    # Get averaged subword embeddings
    subword_embeddings = get_subword_embeddings(word, model, tokenizer)
    avg_subword_embedding = torch.mean(subword_embeddings, dim=0)

    # Calculate similarity
    similarity = cosine_similarity(whole_word_embedding.unsqueeze(0), avg_subword_embedding.unsqueeze(0)).item()

    # Frequency-based boost or penalty
    frequency_boost = 1.0
    if word in frequency_dict:
        frequency = frequency_dict[word]
        if frequency > 50000:
            frequency_boost = 2.0
        elif frequency > 10000:
            frequency_boost = 1.5
        else:
            frequency_boost = 0.8

    complexity_score = similarity * frequency_boost
    return complexity_score

def compute_mlm_score(sentence, model, tokenizer, frequency_dict):
    tokens = tokenizer(sentence, return_tensors="pt").to(device)
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
        complexity_score = compute_complexity_score(word, model, tokenizer, frequency_dict)
        penalized_score = score * complexity_score
        
        scores.append(penalized_score)

    average_score = sum(scores) / len(scores) * 100
    return average_score, scores

def compute_word_score(word, sentence, model, tokenizer, frequency_dict):
    words = sentence.split()
    if word not in words:
        raise ValueError(f"The word '{word}' is not found in the sentence.")

    index = words.index(word)
    sub_sentence = ' '.join(words[:index + 1])

    # Tokenize and mask the word
    tokens = tokenizer(sub_sentence, return_tensors="pt").to(device)
    masked_input_ids = tokens['input_ids'].clone()
    word_token_index = tokens['input_ids'][0].size(0) - 2
    masked_input_ids[0, word_token_index] = tokenizer.mask_token_id

    # Get MLM prediction probabilities
    with torch.no_grad():
        outputs = model(masked_input_ids)
    logits = outputs.logits
    word_token_id = tokens['input_ids'][0, word_token_index]
    probs = torch.softmax(logits[0, word_token_index], dim=-1)
    score = probs[word_token_id].item() * 100  # Convert to percentage

    # Adjust score based on frequency and thresholds
    frequency = frequency_dict.get(word, 0)
    if frequency > 50000 and score < 50.0:
        score += 40.0
    elif frequency > 10000 and score < 40.0:
        score += 20.0
    elif score < 40.0:
        score -= 30.0

    # Ensure score stays within bounds (0-100)
    score = max(0, min(score, 100))

    # Apply frequency-based complexity score
    complexity_score = compute_complexity_score(word, model, tokenizer, frequency_dict)
    penalized_score = score * complexity_score

    return penalized_score

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

def find_row_containing_string(data, column_name, search_string):
    for row in data:
        if search_string in row[column_name]:
            return row  # Return the first matching row
    return None  # Return None if no match is found

def process_ngram_parallel(ngram_sentence, rough_pos, model, tokenizer, threshold, frequency_dict):
    ngram_sentence = undo_escape_and_wrap(ngram_sentence)
    sequence_mlm_score, _ = compute_mlm_score(ngram_sentence, model, tokenizer, frequency_dict)
    sequence_mlm_score = float(sequence_mlm_score)
    
    if sequence_mlm_score >= threshold:
        comparison_matrix = ['*'] * len(ngram_sentence.split())
        new_pattern = rough_pos.split()
        words = ngram_sentence.split()
        rough_pos_tokens = rough_pos.split()

        if len(words) != len(rough_pos_tokens):
            print("Length mismatch between words and POS tokens for n-gram. Skipping...")
            return None, None, None, False

        all_asterisks = True  # Track if all replacements remain as '*'
        for i, (pos_tag, word) in enumerate(zip(rough_pos_tokens, words)):
            word_score = compute_word_score(word, ngram_sentence, model, tokenizer, frequency_dict)
            if word_score >= threshold:
                new_pattern[i] = word
                comparison_matrix[i] = word
                all_asterisks = False  # A replacement was made
            else:
                new_pattern[i] = pos_tag

        final_hybrid_ngram = ' '.join(new_pattern)
        return final_hybrid_ngram, comparison_matrix, sequence_mlm_score, all_asterisks

    return None, None, None, False

def generalize_patterns_parallel(ngram_list_file, pos_patterns_file, id_array_file, output_file, lexeme_comparison_dict_file, model, tokenizer, frequency_dict, threshold=80.0, start_pattern_id=None):
    print("Loading POS patterns...")
    pos_patterns = load_csv(pos_patterns_file)

    print("Loading ID arrays")
    id_array = load_csv(id_array_file)

    print("Loading ngram list file")
    ngram_list = load_csv(ngram_list_file)

    seen_lexeme_comparisons = load_lexeme_comparison_dictionary(lexeme_comparison_dict_file)
    
    latest_pattern_id_input = get_latest_pattern_id(pos_patterns_file)
    latest_pattern_id_output = get_latest_pattern_id(output_file)
    latest_pattern_id = max(latest_pattern_id_input, latest_pattern_id_output)
    
    pattern_counter = latest_pattern_id + 1
    pos_comparison_results = []

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'POS_N-Gram', 'Lexeme_N-Gram', 'MLM_Scores', 'Comparison_Replacement_Matrix', 'Final_Hybrid_N-Gram']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            for pos_pattern in tqdm(pos_patterns, desc="POS Patterns"):
                pattern_id = pos_pattern['Pattern_ID']

                # Skip patterns until the starting pattern ID is reached (if specified)
                if start_pattern_id and int(pattern_id) < int(start_pattern_id):
                    continue

                pattern = pos_pattern['POS_N-Gram']
                writer.writerow({
                    'Pattern_ID': pattern_id,
                    'POS_N-Gram': pattern,
                    'Lexeme_N-Gram': '',
                    'MLM_Scores': '',
                    'Comparison_Replacement_Matrix': '',
                    'Final_Hybrid_N-Gram': pattern
                })

                id_array_row = find_row_containing_string(id_array, 'Pattern_ID', pattern_id)
                if id_array_row is None:
                    print(f"No matching row found for Pattern_ID: {pattern_id}")
                    continue

                id_array_value = id_array_row.get("ID_Array")

                futures = []
                for instance_id in convert_id_array(id_array_value):
                    for ngram in ngram_list:
                        if ngram['N-Gram_ID'] == instance_id.zfill(6):
                            ngram_sentence = ngram.get('N-Gram', '')
                            futures.append(executor.submit(
                                process_ngram_parallel, ngram_sentence, pattern, model, tokenizer, threshold, frequency_dict
                            ))

                for future in as_completed(futures):
                    hybrid_ngram, comparison_matrix, sequence_mlm_score, is_all_asterisks = future.result()
                    if hybrid_ngram and comparison_matrix:
                        pattern_counter += 1
                        new_pattern_id = generate_pattern_id(pattern_counter)

                        # Add to comparison dictionary
                        seen_lexeme_comparisons[(pattern, ngram_sentence)] = new_pattern_id

                        if not is_all_asterisks:
                            # Only add to output if the comparison matrix isn't all asterisks
                            pos_comparison_results.append({
                                'Pattern_ID': new_pattern_id,
                                'POS_N-Gram': pattern,
                                'Lexeme_N-Gram': redo_escape_and_wrap(ngram_sentence),
                                'MLM_Scores': sequence_mlm_score,
                                'Comparison_Replacement_Matrix': comparison_matrix,
                                'Final_Hybrid_N-Gram': hybrid_ngram
                            })

                writer.writerows(pos_comparison_results)
                pos_comparison_results = []

        save_lexeme_comparison_dictionary(lexeme_comparison_dict_file, seen_lexeme_comparisons)

# Load frequency dictionary before calling the function
frequency_dict = load_frequency_dict('rules/database/word_frequency.csv')

# Call the parallelized version
for n in range(4, 5):
    ngram_list_file = 'rules/database/ngram.csv'
    pos_patterns_file = f'rules/database/Generalized/POSTComparison/{n}grams.csv'
    id_array_file = f'rules/database/POS/{n}grams.csv'
    output_file = f'rules/database/Generalized/LexemeComparison/{n}grams.csv'
    comparison_dict_file = 'rules/database/LexComparisonDictionary.txt'

    # Specify the starting pattern ID if resuming from an interruption
    start_pattern_id = "000000"  # Replace with the desired starting Pattern ID

    print(f"Starting generalization for {n}-grams...")
    generalize_patterns_parallel(ngram_list_file, pos_patterns_file, id_array_file, output_file, comparison_dict_file, roberta_model, roberta_tokenizer, frequency_dict, start_pattern_id=start_pattern_id)
    print(f"Finished generalization for {n}-grams.")
