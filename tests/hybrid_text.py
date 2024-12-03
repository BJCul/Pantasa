import pandas as pd
import re
from collections import defaultdict

# Hierarchical POS Tag Dictionary
hierarchical_pos_tags = {
    "NN.*": ["NNC", "NNP", "NNPA", "NNCA"],
    "PR.*": ["PRS", "PRP", "PRSP", "PRO", "PRQ", "PRQP", "PRL", "PRC", "PRF", "PRI"],
    "DT.*": ["DTC", "DTCP", "DTP", "DTPP"],
    "CC.*": ["CCT", "CCR", "CCB", "CCA", "CCP", "CCU"],
    "LM": [],
    "TS": [],
    "VB.*": ["VBW", "VBS", "VBH", "VBN", "VBTS", "VBTR", "VBTF", "VBTP", "VBAF", "VBOF", "VBOB", "VBOL", "VBOI", "VBRF"],
    "JJ.*": ["JJD", "JJC", "JJCC", "JJCS", "JJCN", "JJN"],
    "RB.*": ["RBD", "RBN", "RBK", "RBP", "RBB", "RBR", "RBQ", "RBT", "RBF", "RBW", "RBM", "RBL", "RBI", "RBJ", "RBS"],
    "CD.*": ["CDB"],
    "FW": [],
    "PM.*": ["PMP", "PME", "PMQ", "PMC", "PMSC", "PMS"]
}

# Expand patterns, keeping exact tokens intact and handling wildcards
def expand_pattern(pattern, pos_dict):
    tokens = pattern.split()
    expanded_patterns = [""]

    for token in tokens:
        if ".*" in token:  # Wildcard POS tag
            possible_tags = pos_dict.get(token, [])
            expanded_patterns = [f"{p} {tag}" for p in expanded_patterns for tag in possible_tags]
        else:  # Exact token or detailed POS tag
            expanded_patterns = [f"{p} {token}" for p in expanded_patterns]
    return [p.strip() for p in expanded_patterns]

# Load the rule pattern bank from ngrams.csv and expand patterns
def load_rule_pattern_bank_from_csv(csv_path, pos_dict):
    """
    Loads the rule pattern bank from `ngrams.csv` using only the Final_Hybrid_N-Gram column.
    Expands patterns with wildcard POS tags.

    Args:
    - csv_path: Path to the `ngrams.csv` file.
    - pos_dict: Hierarchical POS tag dictionary.

    Returns:
    - A list of expanded rules.
    """
    df = pd.read_csv(csv_path)
    expanded_rule_bank = []

    for _, row in df.iterrows():
        final_hybrid_ngram = row["Final_Hybrid_N-Gram"]
        if pd.notna(final_hybrid_ngram):  # Only process non-empty entries
            expanded_patterns = expand_pattern(final_hybrid_ngram, pos_dict)
            for pattern in expanded_patterns:
                expanded_rule_bank.append({"pattern": pattern})

    return expanded_rule_bank

# Weighted Levenshtein Distance Function for detailed, wildcard, and hybrid matching
def edit_weighted_levenshtein(input_ngram, pattern_ngram, pos_dict):
    input_tokens = input_ngram.split()
    pattern_tokens = pattern_ngram.split()
    len_input = len(input_tokens)
    len_pattern = len(pattern_tokens)

    distance_matrix = [[0] * (len_pattern + 1) for _ in range(len_input + 1)]

    for i in range(len_input + 1):
        distance_matrix[i][0] = i
    for j in range(len_pattern + 1):
        distance_matrix[0][j] = j

    substitution_weight = 7.0
    insertion_weight = 8.0
    deletion_weight = 1.2

    for i in range(1, len_input + 1):
        for j in range(1, len_pattern + 1):
            input_token = input_tokens[i - 1]
            pattern_token = pattern_tokens[j - 1]

            # Match detailed or wildcard POS tags
            if pattern_token in pos_dict and input_token in pos_dict[pattern_token]:
                cost = 0
            elif input_token == pattern_token:  # Exact token match
                cost = 0
            else:
                cost = substitution_weight

            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + deletion_weight,  # Deletion
                distance_matrix[i][j - 1] + insertion_weight,  # Insertion
                distance_matrix[i - 1][j - 1] + cost  # Substitution
            )
    return distance_matrix[len_input][len_pattern]

def load_rule_pattern_bank_from_csv(csv_path):
    """
    Loads the rule pattern bank from `ngrams.csv` using only the Final_Hybrid_N-Gram column.
    Does not expand wildcard patterns.
    """
    df = pd.read_csv(csv_path)
    rule_pattern_bank = []

    for _, row in df.iterrows():
        final_hybrid_ngram = row["Final_Hybrid_N-Gram"]
        if pd.notna(final_hybrid_ngram):  # Only process non-empty entries
            rule_pattern_bank.append({"pattern": final_hybrid_ngram})

    return rule_pattern_bank

def match_ngrams_regex(input_ngram, rule_pattern_bank, pos_dict):
    """
    Match input n-grams directly against wildcard patterns using regex.
    
    Args:
    - input_ngram: Input n-gram to match.
    - rule_pattern_bank: List of raw patterns from Final_Hybrid_N-Gram.
    - pos_dict: Hierarchical POS tag dictionary.

    Returns:
    - The best matching pattern and its distance.
    """
    best_match = None
    min_distance = float("inf")

    for rule in rule_pattern_bank:
        raw_pattern = rule["pattern"]

        # Convert wildcard patterns into regex
        regex_pattern = re.escape(raw_pattern).replace(r"\.\*", ".*")
        regex_pattern = "^" + regex_pattern + "$"

        # Match input n-gram directly using regex
        if re.match(regex_pattern, input_ngram):
            # Compute Levenshtein distance for matched patterns
            distance = edit_weighted_levenshtein(input_ngram, raw_pattern, pos_dict)
            if distance < min_distance:
                min_distance = distance
                best_match = raw_pattern

    return best_match, min_distance


# Test the functionality with ngrams.csv
def test_ngrams_csv_rule_matching():
    # Input N-Grams to test
    input_ngrams = [
        "NNC na NNC CCT NNC LM",     # NNC na NN.* CC.* NN.* LM
        "VBAF NNC JJC LM NNCA CCT" # VBTS CC.* JJ.* CC.* NN.* CC.*          
    ]
    # Path to ngrams.csv
    csv_path = "data/processed/6grams.csv"  # Update with the actual path data/processed/detailed_hngram.csv

    # Load expanded rule pattern bank
    expanded_rule_bank = load_rule_pattern_bank_from_csv(csv_path)

    # Test each input n-gram
    for input_ngram in input_ngrams:
        best_match, distance = match_ngrams_regex(input_ngram, expanded_rule_bank, hierarchical_pos_tags)
        print(f"Input N-Gram: {input_ngram}")
        print(f"Best Match: {best_match}")
        print(f"Distance: {distance}\n")

# Run the test
test_ngrams_csv_rule_matching()
