import pandas as pd
from collections import defaultdict

# Step 1: Extract n-grams from sentences (POS-tagged and lemmatized data)
def extract_n_grams(pos_tagged_sentences, n=4):
    """Extract n-grams from POS-tagged sentences."""
    n_grams = []
    for sentence in pos_tagged_sentences:
        tokens = sentence.split()  # Each token is a word_POS pair (e.g., "kumain_VBTS")
        for i in range(len(tokens) - n + 1):
            n_gram = tokens[i:i + n]
            n_grams.append(n_gram)
    return n_grams

# Step 2: Group n-grams by their POS tag patterns
def group_n_grams_by_pos(n_grams):
    """Cluster n-grams by their POS tag sequences."""
    pos_tag_clusters = defaultdict(list)
    
    for n_gram in n_grams:
        pos_tags = [token.split('_')[1] for token in n_gram]  # Extract POS tags
        pos_tag_pattern = ' '.join(pos_tags)
        pos_tag_clusters[pos_tag_pattern].append(n_gram)
    
    return pos_tag_clusters

# Step 3: Convert to Hybrid N-Grams
def convert_to_hybrid_n_grams(pos_tag_clusters, threshold=0.75):
    """Convert n-grams to hybrid n-grams based on the 75% token rule."""
    hybrid_n_grams = {}

    for pos_pattern, n_grams in pos_tag_clusters.items():
        # Only consider clusters with at least 4 members
        if len(n_grams) < 4:
            continue
        
        # For each position in the n-gram, decide whether to retain the token or generalize to POS
        hybrid_n_gram = []
        n_gram_length = len(n_grams[0])

        for i in range(n_gram_length):
            tokens_at_index = [n_gram[i].split('_')[0] for n_gram in n_grams]  # Extract tokens at position i
            most_common_token = max(set(tokens_at_index), key=tokens_at_index.count)
            token_frequency = tokens_at_index.count(most_common_token) / len(tokens_at_index)

            # Retain the token if it appears in 75% or more of the n-grams, else generalize to POS tag
            if token_frequency >= threshold:
                hybrid_n_gram.append(most_common_token)
            else:
                pos_tag = n_grams[0][i].split('_')[1]  # Use the POS tag of the first n-gram
                hybrid_n_gram.append(f"[{pos_tag}]")

        hybrid_n_grams[pos_pattern] = ' '.join(hybrid_n_gram)

    return hybrid_n_grams

# Step 4: Save the hybrid n-grams to a CSV
def save_hybrid_n_grams_to_csv(hybrid_n_grams, output_file):
    """Save hybrid n-grams to a CSV file."""
    hybrid_ngram_list = [{'Pattern_ID': idx + 1, 'Final_Hybrid_N-Gram': hybrid_ngram} 
                         for idx, hybrid_ngram in enumerate(hybrid_n_grams.values())]
    hybrid_ngram_df = pd.DataFrame(hybrid_ngram_list)
    hybrid_ngram_df.to_csv(output_file, index=False)

# Step 5: Full pipeline for generating hybrid n-grams
def generate_hybrid_n_grams(pos_tagged_sentences, output_file, n=4):
    """Generate hybrid n-grams from POS-tagged sentences and save to CSV."""
    n_grams = extract_n_grams(pos_tagged_sentences, n=n)
    pos_tag_clusters = group_n_grams_by_pos(n_grams)
    hybrid_n_grams = convert_to_hybrid_n_grams(pos_tag_clusters)
    save_hybrid_n_grams_to_csv(hybrid_n_grams, output_file)
    print(f"Hybrid N-Grams saved to {output_file}")

# ----------------------- Example Usage ---------------------------

if __name__ == "__main__":
    # Sample POS-tagged sentences (each token is in the format word_POS)
    pos_tagged_sentences = [
        "nagpunta_VBTS sa_CC bahay_NNC",
        "namili_VBTS sa_CC bayan_NNC",
        "kumain_VBTS sa_CC tindahan_NNC",
        "bumili_VBTS sa_CC merkado_NNC"
    ]

    # Path to save the generated hybrid n-grams
    output_file = 'data/processed/hngrams.csv'

    # Generate and save hybrid n-grams
    generate_hybrid_n_grams(pos_tagged_sentences, output_file, n=4)