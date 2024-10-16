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

def subword_penalty_score(score, word, tokenizer):
    # Tokenize the word and count how many subword tokens it splits into
    tokenized_word = tokenizer(word, return_tensors="pt")['input_ids'][0]
    num_tokens = len(tokenized_word)  # Count number of subwords

    # Penalize the score based on the number of subword tokens
    penalized_score = score / num_tokens  # Fewer tokens = higher score
    return penalized_score


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
        
        # Get the word corresponding to the token ID
        word = tokenizer.decode([original_token_id]).strip()

        # Apply subword complexity penalty
        penalized_score = subword_penalty_score(score, word, tokenizer)
        
        scores.append(penalized_score)

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
    
    # Apply subword complexity penalty
    penalized_score = subword_penalty_score(score, word, tokenizer)
    
    return penalized_score * 100  # Return as a percentage

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


def process_ngram(ngram_sentence, rough_pos, model=roberta_model, tokenizer=roberta_tokenizer, threshold=80.0):
    
    # Compute MLM score for the full sequence
    sequence_mlm_score, _ = compute_mlm_score(ngram_sentence, model, tokenizer)
    
    if sequence_mlm_score >= threshold:
        print(f"Sequence MLM score {sequence_mlm_score} meets the threshold {threshold}. Computing individual word scores...")
        
        # Initialize comparison matrix and new pattern
        comparison_matrix = ['*'] * len(ngram_sentence.split())
        new_pattern = rough_pos.split()
        words = ngram_sentence.split()
        rough_pos_tokens = rough_pos.split()

        # Check for length mismatch between words and POS tokens
        if len(words) != len(rough_pos_tokens):
            print("Length mismatch between words and POS tokens for n-gram. Skipping...")
            return None, None

        # Process each word in the sentence
        for i, (pos_tag, word) in enumerate(zip(rough_pos_tokens, words)):
            word_score = compute_word_score(word, ngram_sentence, model, tokenizer)
            print(f"Word '{word}' average score: {word_score}")
            
            if word_score >= threshold:
                new_pattern[i] = word
                comparison_matrix[i] = word
            else:
                new_pattern[i] = pos_tag  # Restore the original POS tag if word doesn't meet threshold

        final_hybrid_ngram = ' '.join(new_pattern)

        return final_hybrid_ngram, comparison_matrix, sequence_mlm_score

def generalize_patterns(ngram_list_file, pos_patterns_file, id_array_file, output_file, lexeme_comparison_dict_file, model, tokenizer,  threshold=80.0):
    print("Loading POS patterns...")
    pos_patterns = load_csv(pos_patterns_file)  # Load full file

    print("Loading ID arrays")
    id_array = load_csv(id_array_file)

    print("Loading ngram list file")
    ngram_list = load_csv(ngram_list_file)

    # Load lexeme comparison dictionary and track existing Pattern_IDs
    seen_lexeme_comparisons = load_lexeme_comparison_dictionary(lexeme_comparison_dict_file)
    
    # Get the latest pattern ID from both input and output files
    latest_pattern_id_input = get_latest_pattern_id(pos_patterns_file)
    latest_pattern_id_output = get_latest_pattern_id(output_file)
    latest_pattern_id = max(latest_pattern_id_input, latest_pattern_id_output)
    
    # Initialize pattern counter from the latest ID
    pattern_counter = latest_pattern_id + 1
    pos_comparison_results = []

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Pattern_ID', 'POS_N-Gram', 'Lexeme_N-Gram', 'MLM_Scores', 'Comparison_Replacement_Matrix', 'Final_Hybrid_N-Gram']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Step 1: Write the hybrid n-grams to the output CSV before comparison
        print("Writing initial hybrid n-grams to the CSV file...")
        process_count = 0
        for pos_pattern in pos_patterns:
            process_count += 1
            pattern_id = pos_pattern['Pattern_ID']
            pattern = pos_pattern['POS_N-Gram']

            # Write the POS n-grams to the output CSV before comparison
            print(f"Listing POS n-gram for Pattern: {pattern_id}")
            writer.writerow({
                'Pattern_ID': pattern_id,
                'POS_N-Gram': pattern,
                'Lexeme_N-Gram': '',
                'MLM_Scores': '',
                'Comparison_Replacement_Matrix': '',
                'Final_Hybrid_N-Gram': pattern_id
            })

            # Step 2: Perform the comparison and update the file
            print(f"Comparing for Pattern: {pattern_id}")
            id_array_row = find_row_containing_string(id_array, 'Pattern_ID', pattern_id)
            if id_array_row is None:
                print(f"No matching row found for Pattern_ID: {pattern_id}")
                continue

            id_array_value = id_array_row.get("ID_Array")
            id_array_total_comparison = len(id_array_value) * 2
            id_array_tally_comparison = 0

            for instance_id in convert_id_array(id_array_value):
                found = False
                for ngram in ngram_list:
                    if ngram['N-Gram_ID'] == instance_id.zfill(6):
                        found = True
                        ngram_sentence = ngram.get('N-Gram', '')
                        comparison_key = (pattern_id, ngram_sentence)
                        if comparison_key not in seen_lexeme_comparisons:
                            score, _ = compute_mlm_score(ngram_sentence, model, tokenizer)
                            if score >= threshold:
                                hybrid_ngram, comparison_matrix = process_ngram(ngram_sentence, pattern)

                                if comparison_matrix is not None:      
                                    # Increment pattern counter and generate new pattern ID
                                    pattern_counter += 1
                                    new_pattern_id = generate_pattern_id(pattern_counter)
                                    
                                    print(f"Compared for instance successful: {instance_id}")
                                    pos_comparison_results.append({
                                        'Pattern_ID': new_pattern_id,  # Use new pattern ID here
                                        'POS_N-Gram': pattern,
                                        'Lexeme_N-Gram': ngram_sentence,
                                        'MLM_Scores': score,
                                        'Comparison_Replacement_Matrix': comparison_matrix,
                                        'Final_Hybrid_N-Gram': hybrid_ngram
                                    })
                                    seen_lexeme_comparisons[comparison_key] = f'1{new_pattern_id}' 

                                else:
                                    seen_lexeme_comparisons[comparison_key] = f'2{instance_id}' 
                                
                            else:
                                seen_lexeme_comparisons[comparison_key] = f'3{instance_id}'  
                        id_array_tally_comparison += 1 

                        lemma_ngram_sentence = ngram.get('Lemma_N-Gram', '')
                        comparison_key = (pattern_id, lemma_ngram_sentence)
                        if comparison_key not in seen_lexeme_comparisons:
                            score, _ = compute_mlm_score(lemma_ngram_sentence, model, tokenizer)
                            if score >= threshold:
                                hybrid_ngram, comparison_matrix = process_ngram(ngram_sentence, pattern)
                                
                                if comparison_matrix is not None:      
                                    # Increment pattern counter and generate new pattern ID
                                    pattern_counter += 1
                                    new_pattern_id = generate_pattern_id(pattern_counter)
                                    
                                    
                                    print(f"Compared for instance successful: {instance_id}")
                                    pos_comparison_results.append({
                                        'Pattern_ID': new_pattern_id,  # Use new pattern ID here
                                        'POS_N-Gram': pattern,
                                        'Lexeme_N-Gram': ngram_sentence,
                                        'MLM_Scores': score,
                                        'Comparison_Replacement_Matrix': comparison_matrix,
                                        'Final_Hybrid_N-Gram': hybrid_ngram
                                    })
                                    seen_lexeme_comparisons[comparison_key] = f'1{new_pattern_id}' 

                                else:
                                    seen_lexeme_comparisons[comparison_key] = f'2{instance_id}' 
                                
                            else:
                                seen_lexeme_comparisons[comparison_key] = f'3{instance_id}'  
                        id_array_tally_comparison += 1

                if id_array_tally_comparison % 10 == 0:
                    print(f"Total comparisons: {id_array_tally_comparison}/{id_array_total_comparison}")

                if not found:
                    print(f"No instance found for ID: {instance_id}")
            

            writer.writerows(pos_comparison_results)
            pos_comparison_results = []  # Reset after saving

        # Save the lexeme comparison dictionary after the entire process
        save_lexeme_comparison_dictionary(lexeme_comparison_dict_file, seen_lexeme_comparisons)

for n in range(6, 7):
    ngram_list_file = 'rules/database/ngrams.csv'
    pos_patterns_file = f'rules/database/Generalized/POSTComparison/{n}grams.csv'
    id_array_file = f'rules/database/POS/{n}grams.csv'
    output_file = f'rules/database/Generalized/LexemeComparison/{n}grams.csv'
    comparison_dict_file = 'rules/database/LexComparisonDictionary.txt'
    batch_size=2

    print(f"Starting generalization for {n}-grams...")
    generalize_patterns(ngram_list_file, pos_patterns_file, id_array_file, output_file, comparison_dict_file, roberta_model, roberta_tokenizer)
    print(f"Finished generalization for {n}-grams.")