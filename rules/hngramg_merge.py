import pandas as pd
import re
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract the n-gram size from the pattern ID
def get_ngram_size_from_pattern_id(pattern_id):
    return int(str(pattern_id)[0])

# Function to filter n-grams based on the n-gram size
def filter_by_ngram_size(ngrams_df, pattern_ngram_size):
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

detailed_taglist = hierarchical_pos_tags.values()

def tag_type(tag):
    # Check if the tag is a combined rough POS tag (i.e., "NN.*_VB.*")
    if "_" in tag:
        components = tag.split("_")  # Split by underscore
        # Check if all components are rough POS tags
        if all(component in hierarchical_pos_tags for component in components):
            return "rough POS tag"
        elif all(component in hierarchical_pos_tags.values() for component in components):
            return "detailed POS tag"
        else:
            return "word"
    # Check if the tag is a rough POS tag
    elif tag in hierarchical_pos_tags:
        return "rough POS tag"
    # Check if the tag is a detailed POS tag (found in the values of the dictionary)
    elif tag in detailed_taglist:
        return "detailed POS tag"
    # Tag not in the heirarchy is a word
    else:
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
            word_pattern.append(r'.*')  # Replace detailed POS with wildcard

        # Detailed POS Pattern
        elif tag_category == "detailed POS tag":
            rough_pos_pattern.append(r'.*')  # Replace rough POS with wildcard
            detailed_pos_pattern.append(part)  # Keep detailed POS tag
            word_pattern.append(r'.*')  # Replace detailed POS with wildcard
        
        # Both POS Pattern
        elif tag_category == "both POS tag":
            rough_pos_pattern.append(part)  # Replace rough POS with wildcard
            detailed_pos_pattern.append(part)  # Replace detailed POS with wildcard
            word_pattern.append(r'.*')  # Replace detailed POS with wildcard

        elif tag_category == "word":
            rough_pos_pattern.append(r'.*')  # Replace rough POS with wildcard
            detailed_pos_pattern.append(r'.*')  # Replace detailed POS with wildcard
            word_pattern.append(part)  # Replace detailed POS with wildcard
               
    # Join each pattern list to form a regex search pattern
    rough_pos_search_pattern = " ".join(rough_pos_pattern)
    detailed_pos_search_pattern = " ".join(detailed_pos_pattern)
    word_search_pattern = " ".join(word_pattern)

    logging.debug(f"Rough POS pattern: {rough_pos_search_pattern}")
    logging.debug(f"Detailed POS pattern: {detailed_pos_search_pattern}")
    logging.debug(f"Word pattern: {word_search_pattern}")
    
    return rough_pos_search_pattern, detailed_pos_search_pattern, word_search_pattern

# Function to apply rough POS, detailed POS, and word-based filtering
def instance_collector(pattern, ngrams_df, pattern_ngram_size):
    logging.debug(f"Searching n-gram matches for pattern id: {pattern}")
    
    # Step 1: Filter by n-gram size
    size_filtered_df = filter_by_ngram_size(ngrams_df, pattern_ngram_size)

    # Step 2: Get the three search patterns (rough POS, detailed POS, and words)
    rough_pos_search_pattern, detailed_pos_search_pattern, word_search_pattern = search_pattern_conversion_based_on_tag_type(pattern)
    
    # Step 3: Apply rough POS filtering
    rough_pos_matches = size_filtered_df[size_filtered_df['RoughPOS_N-Gram'].str.contains(rough_pos_search_pattern, regex=True)]
    
    # Step 4: Apply detailed POS filtering on the rough POS matches
    detailed_pos_matches = rough_pos_matches[rough_pos_matches['DetailedPOS_N-Gram'].str.contains(detailed_pos_search_pattern, regex=True)]

    # Step 5: Apply word filtering on thhe detailed POS matches
    word_matches = detailed_pos_matches[detailed_pos_matches['N-Gram'].str.contains(word_search_pattern, regex=True)]
    
    return word_matches


# Function to build hngrams from lexeme and n-gram files
def build_hngrams(lexeme_file, ngram_list_file, hngram_file):
    # Load the lexeme and n-gram files
    lexeme_data = pd.read_csv(lexeme_file)
    ngram_data = pd.read_csv(ngram_list_file)

    # Prepare a DataFrame for storing the output
    hngram_data = pd.DataFrame()

    # Process each row in the lexeme file
    for index, row in lexeme_data.iterrows():
        pattern = row['Final_Hybrid_N-Gram']
        pattern_id = row['Pattern_ID']

        # Generate Detailed_POS, Rough_POS, and Lexeme using the search function
        rough_pos, detailed_pos, lexeme = search_pattern_conversion_based_on_tag_type(pattern)

        # Determine the n-gram size from the pattern ID
        ngram_size = get_ngram_size_from_pattern_id(pattern_id)

        # Use the instance_collector to find matching n-grams
        matches = instance_collector(pattern, ngram_data, ngram_size)
        print(matches)

        # Collect IDs and frequency
        id_array = matches['N-Gram_ID'].tolist()
        frequency = len(id_array)

        # Skip rows with frequency 0
        if frequency == 0:
            logging.debug(f"Skipping pattern '{pattern}' with frequency 0")
            continue

        # Append the row to the hngram_data DataFrame
        hngram_data = hngram_data._append({
            'Pattern_ID': pattern_id,
            'Hybrid_N-Gram': pattern,
            'Rough_POS': rough_pos,
            'Detailed_POS': detailed_pos,
            'Lexeme': lexeme,
            'Frequency': frequency,
            'ID_Array': id_array

        }, ignore_index=True)

    # Save the combined results to the hngram_file
    hngram_data.to_csv(hngram_file, mode='a', header=False, index=False)

# Loop through n-gram sizes and process the files
for n in range(6, 8):  # Adjust range as needed
    ngram_list_file = 'rules/database/hng_ngrams.csv'
    lexeme_file = f'rules/database/Generalized/LexemeComparison/{n}grams.csv'
    hngram_file = "rules/database/hngrams.csv"

    # Build the hngrams
    build_hngrams(lexeme_file, ngram_list_file, hngram_file)