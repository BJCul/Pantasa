import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.nn.functional import cosine_similarity
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define the model
global_max_score = 0
global_min_score = 0
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"
print("Loading tokenizer...")
roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)
print("Loading model...")
roberta_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)
print("Model and tokenizer loaded successfully.")

# Load frequency dictionary for word frequency-based boost/penalty
def load_frequency_dict(file_path):
    frequency_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word, frequency = line.strip().split(',')
            frequency_dict[word] = int(frequency)
    return frequency_dict

# Load the frequency dictionary (assumes a CSV with 'word,frequency' format)
frequency_dict = load_frequency_dict("path/to/frequency_dict.csv")

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
        return []
    return id_array_str.strip("[]'").replace("'", "").split(', ')

def load_and_convert_csv(file_path):
    data = load_csv(file_path)
    for entry in data:
        print(f"Processing entry: {entry}")
        entry['ID_Array'] = convert_id_array(entry.get('ID_Array', ''))
    return data

def get_subword_embeddings(word, model, tokenizer):
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    subword_embeddings = outputs.hidden_states[-1][0][1:-1]  # Ignore CLS and SEP tokens
    return subword_embeddings

def compute_complexity_score(word, model, tokenizer, frequency_dict):
    # Get the whole word embedding
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    whole_word_embedding = torch.mean(outputs.hidden_states[-1][0][1:-1], dim=0)

    # Get averaged subword embeddings
    subword_embeddings = get_subword_embeddings(word, model, tokenizer)
    avg_subword_embedding = torch.mean(subword_embeddings, dim=0)

    # Calculate similarity
    similarity = cosine_similarity(whole_word_embedding, avg_subword_embedding, dim=0).item()

    # Frequency-based boost or penalty
    frequency_boost = 1.0
    if word in frequency_dict:
        frequency = frequency_dict[word]
        if frequency > 50000:
            frequency_boost = 1.5
        elif frequency > 1000:
            frequency_boost = 1.0
        else:
            frequency_boost = 0.8

    complexity_score = similarity * frequency_boost
    return complexity_score

def compute_mlm_score(sentence, model, tokenizer, frequency_dict):
    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    scores = []

    for i in range(1, input_ids.size(0) - 1):  # Skip [CLS] and [SEP] tokens
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

    average_score = sum(scores) / len(scores) * 100  # Convert to percentage
    return average_score, scores

def compute_word_score(word, sentence, model, tokenizer, frequency_dict):
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

    complexity_score = compute_complexity_score(word, model, tokenizer, frequency_dict)
    penalized_score = score * complexity_score

    return penalized_score * 100

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
    """
    This function allows parallel processing of n-grams.
    """
    # Compute MLM score for the full sequence
    sequence_mlm_score, _ = compute_mlm_score(ngram_sentence, model, tokenizer, frequency_dict)
    
    if sequence_mlm_score >= threshold:
        print(f"Sequence MLM score {sequence_mlm_score} meets the threshold {threshold}. Computing individual word scores...")
        
        comparison_matrix = ['*'] * len(ngram_sentence.split())
        new_pattern = rough_pos.split()
        words = ngram_sentence.split()
        rough_pos_tokens = rough_pos.split()

        if len(words) != len(rough_pos_tokens):
            print("Length mismatch between words and POS tokens for n-gram. Skipping...")
            return None, None, None

        for i, (pos_tag, word) in enumerate(zip(rough_pos_tokens, words)):
            word_score = compute_word_score(word, ngram_sentence, model, tokenizer, frequency_dict)
            if word_score >= threshold:
                new_pattern[i] = word
                comparison_matrix[i] = word
            else:
                new_pattern[i] = pos_tag

        final_hybrid_ngram = ' '.join(new_pattern)
        return final_hybrid_ngram, comparison_matrix, sequence_mlm_score

    return None, None, None

def generalize_patterns_parallel(ngram_list_file, pos_patterns_file, id_array_file, output_file, lexeme_comparison_dict_file, model, tokenizer, frequency_dict, threshold=80.0):
    print("Loading POS patterns...")
    pos_patterns = load_csv(pos_patterns_file)

    print("Loading ID arrays")
    id_array = load_csv(id_array_file)

    print("Loading ngram list file")
    ngram_list = load_csv(ngram_list_file)

    seen_ngrams = load_existing_results(output_file)
    latest_pattern_id = get_latest_pattern_id(output_file) + 1

    new_lexeme_comparison_dict = load_lexeme_comparison_dictionary(lexeme_comparison_dict_file)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for ngram_data in ngram_list:
            rough_pos = ngram_data['POS']  # Or adjust based on your data's key
            sentence = ngram_data['N-Gram']  # Adjust based on your data's key

            futures.append(executor.submit(process_ngram_parallel, sentence, rough_pos, model, tokenizer, threshold, frequency_dict))

        for future in tqdm(as_completed(futures), total=len(futures)):
            final_ngram, comparison_matrix, mlm_score = future.result()
            if final_ngram and final_ngram not in seen_ngrams:
                seen_ngrams.add(final_ngram)
                pattern_id = generate_pattern_id(latest_pattern_id)
                latest_pattern_id += 1
                row = {
                    'Pattern_ID': pattern_id,
                    'Final_Hybrid_N-Gram': final_ngram,
                    'POS_Pattern_Comparison': ', '.join(comparison_matrix),
                    'Lexeme_Comparison_ID': new_lexeme_comparison_dict.get((rough_pos, final_ngram), ""),
                    'MLM_Score': mlm_score
                }
                with open(output_file, 'a', encoding='utf-8', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=row.keys())
                    if file.tell() == 0:  # Check if file is empty to write headers
                        writer.writeheader()
                    writer.writerow(row)
                print(f"Saved: {row}")

# Example usage of the generalize_patterns_parallel function
ngram_list_file = "path/to/ngram_list.csv"
pos_patterns_file = "path/to/pos_patterns.csv"
id_array_file = "path/to/id_array.csv"
output_file = "path/to/output_file.csv"
lexeme_comparison_dict_file = "path/to/lexeme_comparison_dict.csv"
generalize_patterns_parallel(ngram_list_file, pos_patterns_file, id_array_file, output_file, lexeme_comparison_dict_file, roberta_model, roberta_tokenizer, frequency_dict)
