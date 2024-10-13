import pandas as pd

# Step 1: Load and create the rule pattern bank
file_path = 'data/processed/hngrams.csv'  # Update with actual path
hybrid_ngrams_df = pd.read_csv(file_path)

# Create a dictionary to store the Rule Pattern Bank (Hybrid N-Grams + Predefined Rules)
rule_pattern_bank = {}

# Store the hybrid n-grams from the CSV file into the rule pattern bank
for index, row in hybrid_ngrams_df.iterrows():
    pattern_id = row['Pattern_ID']
    hybrid_ngram = row['Final_Hybrid_N-Gram']
    rule_pattern_bank[pattern_id] = {'hybrid_ngram': hybrid_ngram}

# Define the Predefined Rules as part of the rule pattern bank (manually curated)
predefined_rules = {
    300001: {'rule': 'Use na/ng based on vowel ending', 'example': 'pagkain na -> pagkaing'},
    300002: {'rule': 'Unmerge mas from verbs', 'example': 'masmalakas -> mas malakas'},
    300003: {'rule': 'Remove incorrect hyphenation', 'example': 'pinaka-matalino -> pinakamatalino'},
    300004: {'rule': 'Merge incorrectly split affixes', 'example': 'pinag sikapan -> pinagsikapan'}
}

# Add predefined rules to the rule pattern bank
for rule_id, rule_data in predefined_rules.items():
    rule_pattern_bank[rule_id] = rule_data

# Step 2: Define the weighted Levenshtein distance function
def weighted_levenshtein(input_ngram, pattern_ngram):
    input_tokens = input_ngram.split()
    pattern_tokens = pattern_ngram.split()
    
    len_input = len(input_tokens)
    len_pattern = len(pattern_tokens)
    
    # Create a matrix to store the edit distances
    distance_matrix = [[0] * (len_pattern + 1) for _ in range(len_input + 1)]
    
    # Initialize base case values (costs of insertions and deletions)
    for i in range(len_input + 1):
        distance_matrix[i][0] = i
    for j in range(len_pattern + 1):
        distance_matrix[0][j] = j

    # Define weights for substitution, insertion, and deletion
    substitution_weight = 0.8
    insertion_weight = 1.0
    deletion_weight = 1.0

    # Compute the distances
    for i in range(1, len_input + 1):
        for j in range(1, len_pattern + 1):
            if input_tokens[i - 1] == pattern_tokens[j - 1]:  # No change required if tokens match
                cost = 0
            else:
                cost = substitution_weight  # Apply substitution weight if tokens differ

            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + deletion_weight,    # Deletion
                distance_matrix[i][j - 1] + insertion_weight,  # Insertion
                distance_matrix[i - 1][j - 1] + cost           # Substitution
            )
    
    return distance_matrix[len_input][len_pattern]

# Step 3: Function to generate correction token tags based on the Levenshtein distance
def generate_correction_tags(input_ngram, pattern_ngram):
    input_tokens = input_ngram.split()
    pattern_tokens = pattern_ngram.split()
    
    tags = []
    
    for i, (input_token, pattern_token) in enumerate(zip(input_tokens, pattern_tokens)):
        if input_token != pattern_token:
            if input_token.startswith(pattern_token[:2]):  # For substitution errors (same POS group)
                tags.append(f'REPLACE_{input_token}_{pattern_token}')
            else:  # General substitution (different POS group)
                tags.append(f'SUBSTITUTE_{input_token}_{pattern_token}')
    
    # Handle if there are extra tokens in the input or pattern
    if len(input_tokens) > len(pattern_tokens):
        tags.append(f'DELETE_{input_tokens[len(pattern_tokens):]}')
    elif len(pattern_tokens) > len(input_tokens):
        tags.append(f'INSERT_{pattern_tokens[len(input_tokens):]}')
    
    return tags

# Improve matching logic to ignore minor variations
def is_valid_match(input_ngram, pattern_ngram):
    """
    Checks if the input n-gram is a valid match for the pattern n-gram,
    ignoring small variations like tense or pluralization that don't need correction.
    """
    input_tokens = input_ngram.split()
    pattern_tokens = pattern_ngram.split()
    
    if len(input_tokens) != len(pattern_tokens):
        return False

    # Example logic: Ignore substitutions where both tokens are verbs, or both are nouns
    for input_token, pattern_token in zip(input_tokens, pattern_tokens):
        if input_token != pattern_token:
            if input_token.startswith('VB') and pattern_token.startswith('VB'):
                continue  # Ignore differences between verb forms
            if input_token.startswith('NN') and pattern_token.startswith('NN'):
                continue  # Ignore differences between noun forms
            return False  # Significant mismatch found

    return True  # No significant mismatches found

# Modify process_input_ngram to skip suggestions for minor variations
def process_input_ngram(input_ngram, threshold=3.0):
    corrections = []
    min_distance = float('inf')
    best_match = None
    best_pattern_id = None

    for pattern_id, pattern_data in rule_pattern_bank.items():
        pattern_ngram = pattern_data.get('hybrid_ngram')
        if pattern_ngram:
            # Check for exact or valid match before computing distance
            if is_valid_match(input_ngram, pattern_ngram):
                print("Valid match found. No corrections needed.")
                return []  # No corrections needed for valid matches

            # Compute Levenshtein distance if no valid match is found
            distance = weighted_levenshtein(input_ngram, pattern_ngram)
            if distance < min_distance:
                min_distance = distance
                best_match = pattern_ngram
                best_pattern_id = pattern_id

    # Only suggest corrections if the distance is below the threshold
    if best_match and min_distance <= threshold:
        correction_tags = generate_correction_tags(input_ngram, best_match)
        corrections.append({
            'pattern_id': best_pattern_id,
            'distance': min_distance,
            'correction_tags': correction_tags
        })
    else:
        print("No corrections needed. Sentence seems valid.")

    return corrections


