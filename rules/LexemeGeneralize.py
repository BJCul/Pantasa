import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch
import os
from torch.nn.functional import cosine_similarity

# Define the model
global_max_score = 0
global_min_score = 0
tagalog_roberta_model = "jcblaise/roberta-tagalog-base"
print("Loading tokenizer...")
roberta_tokenizer = AutoTokenizer.from_pretrained(tagalog_roberta_model)
print("Loading MLM model...")
roberta_mlm_model = AutoModelForMaskedLM.from_pretrained(tagalog_roberta_model)
print("Loading model for embedding retrieval...")
roberta_embedding_model = AutoModel.from_pretrained(tagalog_roberta_model)
print("Model and tokenizer loaded successfully.")

def load_csv(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    print(f"Data loaded from {file_path}. Number of rows: {len(data)}")
    return data

# New function: Retrieve subword embeddings and compute average
def get_subword_embeddings(word):
    tokens = roberta_tokenizer(word, return_tensors="pt")
    outputs = roberta_embedding_model(**tokens, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # Last layer
    subword_embeddings = hidden_states[0][1:-1]  # Exclude [CLS] and [SEP] tokens
    return subword_embeddings

def average_subword_embedding(word):
    subword_embeddings = get_subword_embeddings(word)
    avg_embedding = torch.mean(subword_embeddings, dim=0)
    return avg_embedding

# Updated complexity score using FastText-like cosine similarity
def compute_complexity_score(word):
    tokens = roberta_tokenizer(word, return_tensors="pt")
    single_word_embedding = roberta_embedding_model(**tokens, output_hidden_states=True).hidden_states[-1][0][1:-1]
    whole_word_embedding = torch.mean(single_word_embedding, dim=0)
    avg_subword_emb = average_subword_embedding(word)

    # Cosine similarity as complexity measure
    similarity = cosine_similarity(whole_word_embedding, avg_subword_emb, dim=0).item()
    boost_factor = 1.2 if similarity > 0.8 else 1.0  # Boost simpler words
    penalized_score = similarity * boost_factor
    return penalized_score

# Updated subword penalty function
def subword_penalty_score(score, word):
    complexity_score = compute_complexity_score(word)  # Calculate complexity
    penalized_score = score * complexity_score  # Apply complexity as a multiplier
    return penalized_score

# MLM scoring function with subword penalty applied
def compute_mlm_score(sentence, model, tokenizer):
    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    scores = []
    
    for i in range(1, input_ids.size(0) - 1):  # Skip [CLS] and [SEP]
        masked_input_ids = input_ids.clone()
        masked_input_ids[i] = tokenizer.mask_token_id  # Mask current token

        with torch.no_grad():
            outputs = model(masked_input_ids.unsqueeze(0))  # Add batch dimension

        logits = outputs.logits[0, i]
        probs = torch.softmax(logits, dim=-1)

        original_token_id = input_ids[i]
        score = probs[original_token_id].item()
        
        # Get word corresponding to the token ID
        word = tokenizer.decode([original_token_id]).strip()
        penalized_score = subword_penalty_score(score, word)  # Apply subword penalty
        
        scores.append(penalized_score)

    average_score = sum(scores) / len(scores) * 100  # Convert to percentage
    return average_score, scores

# Main n-gram processing with updated MLM scoring
def process_ngram(ngram_sentence, rough_pos, model=roberta_mlm_model, tokenizer=roberta_tokenizer, threshold=80.0):
    sequence_mlm_score, _ = compute_mlm_score(ngram_sentence, model, tokenizer)
    
    if sequence_mlm_score >= threshold:
        print(f"Sequence MLM score {sequence_mlm_score} meets the threshold {threshold}. Computing individual word scores...")

        comparison_matrix = ['*'] * len(ngram_sentence.split())
        new_pattern = rough_pos.split()
        words = ngram_sentence.split()
        rough_pos_tokens = rough_pos.split()

        if len(words) != len(rough_pos_tokens):
            print("Length mismatch between words and POS tokens for n-gram. Skipping...")
            return None, None

        for i, (pos_tag, word) in enumerate(zip(rough_pos_tokens, words)):
            word_score = compute_word_score(word, ngram_sentence, model, tokenizer)
            print(f"Word '{word}' average score: {word_score}")
            
            if word_score >= threshold:
                new_pattern[i] = word
                comparison_matrix[i] = word
            else:
                new_pattern[i] = pos_tag  # Restore original POS tag if below threshold

        final_hybrid_ngram = ' '.join(new_pattern)
        return final_hybrid_ngram, comparison_matrix, sequence_mlm_score

# The remaining functions for file I/O and CSV handling stay the same.
