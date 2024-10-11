import pandas as pd
import re
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract the n-gram size from the pattern ID
def get_ngram_size_from_pattern_id(pattern_id):
    return int(str(pattern_id)[0])

# Function to filter n-grams based on the n-gram size
def filter_by_ngram_size(pattern, ngrams_df, pattern_ngram_size):
    logging.debug(f"Filtering n-grams based on size: {pattern_ngram_size}")
    return ngrams_df[ngrams_df['N-Gram_Size'] == pattern_ngram_size]

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

# Function to determine if a tag is a rough POS tag, detailed POS tag, or a word
def tag_type(tag):
    # Check if the tag is a rough POS tag
    if tag in hierarchical_pos_tags:
        return "rough POS tag"
    
    # Check if the tag is a combined detailed POS tag (i.e., "PMS_NNC_CCP")
    if "_" in tag:
        components = tag.split("_")
        if all(any(component in detailed_tags for detailed_tags in hierarchical_pos_tags.values()) for component in components):
            return "detailed POS tag"
    
    # Check if the tag is a detailed POS tag (found in the values of the dictionary)
    for rough_tag, detailed_tags in hierarchical_pos_tags.items():
        if tag in detailed_tags:
            return "detailed POS tag"
    
    # If the tag is neither a rough POS nor a detailed POS tag, it's considered a word
    return "word"


# Function to return 3 patterns based on rough POS tags, detailed POS tags, and words
def search_pattern_conversion_based_on_tag_type(pattern):
    logging.debug(f"Original pattern: {pattern}")
    
    pattern_parts = pattern.split()
    
    # Separate patterns for rough POS, detailed POS, and words
    rough_pos_pattern = []
    detailed_pos_pattern = []
    word_pattern = []

    for part in pattern_parts:
        tag_category = tag_type(part)
        
        # Rough POS Pattern
        if tag_category == "rough POS tag":
            rough_pos_pattern.append(part)  # Keep rough POS tag
            detailed_pos_pattern.append(r'.*')  # Replace detailed POS with wildcard
            word_pattern.append(r'.*')  # Replace words with wildcard
        # Detailed POS Pattern
        elif tag_category == "detailed POS tag":
            rough_pos_pattern.append(r'.*')  # Replace rough POS with wildcard
            detailed_pos_pattern.append(part)  # Keep detailed POS tag
            word_pattern.append(r'.*')  # Replace words with wildcard
        # Word Pattern
        else:
            rough_pos_pattern.append(r'.*')  # Replace rough POS with wildcard
            detailed_pos_pattern.append(r'.*')  # Replace detailed POS with wildcard
            word_pattern.append(part) 
    
    # Join each pattern list to form a regex search pattern
    rough_pos_search_pattern = " ".join(rough_pos_pattern)
    detailed_pos_search_pattern = " ".join(detailed_pos_pattern)
    word_search_pattern = " ".join(word_pattern)

    logging.debug(f"Rough POS pattern: {rough_pos_search_pattern}")
    logging.debug(f"Detailed POS pattern: {detailed_pos_search_pattern}")
    logging.debug(f"Word pattern: //{word_search_pattern}//")
    
    return rough_pos_search_pattern, detailed_pos_search_pattern, word_search_pattern

# Function to apply rough POS, detailed POS, and word-based filtering
def instance_collector(pattern, ngrams_df, pattern_ngram_size):
    logging.debug(f"Searching n-gram matches for pattern id: {pattern}")
    
    # Step 1: Filter by n-gram size
    size_filtered_df = filter_by_ngram_size(pattern, ngrams_df, pattern_ngram_size)
    logging.debug(f"Size-filtered n-grams: {len(size_filtered_df)}")

    # Step 2: Get the three search patterns (rough POS, detailed POS, and words)
    rough_pos_search_pattern, detailed_pos_search_pattern, word_search_pattern = search_pattern_conversion_based_on_tag_type(pattern)
    
    # Step 3: Apply rough POS filtering
    rough_pos_matches = size_filtered_df[size_filtered_df['RoughPOS_N-Gram'].str.contains(rough_pos_search_pattern, regex=True)]
    logging.debug(f"Rough POS tag matches: {len(rough_pos_matches)}")
    
    # Step 4: Apply detailed POS filtering on the rough POS matches
    detailed_pos_matches = rough_pos_matches[rough_pos_matches['DetailedPOS_N-Gram'].str.contains(detailed_pos_search_pattern, regex=True)]
    logging.debug(f"Detailed POS tag matches: {len(detailed_pos_matches)}")
    
    
    # Step 5: Apply word-based filtering using re.search
    def word_match_search(row, word_search_pattern):
        match = re.search(word_search_pattern, row['N-Gram'], re.IGNORECASE)
        logging.debug(f"Testing regex on N-Gram: {row['N-Gram']} | Match: {match}")
        return bool(match)

    final_matches = detailed_pos_matches[detailed_pos_matches.apply(lambda row: word_match_search(row, word_search_pattern), axis=1)]
    logging.debug(f"Word matches: {len(final_matches)}")
    
    
    return final_matches


# Example usage
# Load the CSV files containing the patterns and n-grams
hngrams_df = pd.read_csv('rules/database/hngrams.csv')
ngrams_df = pd.read_csv('rules/database/ngrams.csv')

# Extract the pattern for Pattern ID 300189
pattern_id = 600228
pattern_row = hngrams_df[hngrams_df['Pattern_ID'] == pattern_id]

if not pattern_row.empty:
    pattern = pattern_row['Final_Hybrid_N-Gram'].values[0]
    pattern_ngram_size = get_ngram_size_from_pattern_id(pattern_id)
    
    # Apply the filtering process, with ".*" at the third index position for rough POS
    final_filtered_ngrams = instance_collector(pattern, ngrams_df, pattern_ngram_size)
    
    # Display the results
    total_matched_ngrams = final_filtered_ngrams.shape[0]
    logging.info(f'Total matched n-grams for Pattern ID {pattern_id}: {total_matched_ngrams}')
    print(final_filtered_ngrams[['N-Gram', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'Lemma_N-Gram']].head(10))
else:
    logging.error(f'Pattern ID {pattern_id} not found.')