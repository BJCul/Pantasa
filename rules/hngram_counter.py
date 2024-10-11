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
    logging.debug(f"Word pattern: {word_search_pattern}")
    
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
        return bool(match)

    final_matches = detailed_pos_matches[detailed_pos_matches.apply(lambda row: word_match_search(row, word_search_pattern), axis=1)]
    logging.debug(f"Word matches: {len(final_matches)}")
    
    return final_matches


# Function to update hngrams_df with batch results and save to CSV
def update_hngrams_csv(hngrams_df, batch_df):
    logging.debug(f"Updating hngrams_df with batch results.")
    
    # Update the original hngrams_df with the processed batch data
    for index, row in batch_df.iterrows():
        pattern_id = row['Pattern_ID']
        frequency = row['Frequency']
        rough_pos_pattern = row['Rough POS']
        
        # Update the corresponding row in the main hngrams_df DataFrame
        hngrams_df.loc[hngrams_df['Pattern_ID'] == pattern_id, 'Frequency'] = frequency
        hngrams_df.loc[hngrams_df['Pattern_ID'] == pattern_id, 'Rough POS'] = rough_pos_pattern
    
    # Save the updated hngrams_df to the CSV file after processing each batch
    hngrams_df.to_csv('rules/database/hngrams.csv', index=False)
    logging.debug(f"hngrams.csv saved after processing batch.")

# Function to process hngrams_df in batches
def process_in_batches(hngrams_df, ngrams_df, batch_size=100):
    total_rows = len(hngrams_df)
    
    # Iterate through the DataFrame in batches
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        logging.info(f"Processing batch from row {start} to {end}")
        
        batch_df = hngrams_df.iloc[start:end].copy()
        
        # Process each pattern in the current batch
        for index, row in batch_df.iterrows():
            pattern_id = row['Pattern_ID']
            pattern = row['Hybrid_N-Gram']
            pattern_ngram_size = get_ngram_size_from_pattern_id(pattern_id)
            
            # Apply the filtering process for each pattern
            final_filtered_ngrams = instance_collector(pattern, ngrams_df, pattern_ngram_size)
            
            # Calculate the frequency of matches
            total_matched_ngrams = final_filtered_ngrams.shape[0]
            logging.info(f'Total matched n-grams for Pattern ID {pattern_id}: {total_matched_ngrams}')
            
            # Get the rough POS pattern for storage
            rough_pos_pattern, _, _ = search_pattern_conversion_based_on_tag_type(pattern)
            
            # Update the batch DataFrame with frequency and rough POS pattern
            batch_df.loc[index, 'Frequency'] = total_matched_ngrams
            batch_df.loc[index, 'Rough POS'] = rough_pos_pattern

        # Update the main hngrams_df with the processed batch and save results to CSV
        update_hngrams_csv(hngrams_df, batch_df)

logging.info("All batches processed and hngrams.csv updated.")


# Load the CSV files containing the patterns and n-grams
hngrams_df = pd.read_csv('rules/database/hngrams.csv')
ngrams_df = pd.read_csv('rules/database/ngrams.csv')

# Process the hngrams.csv in batches of 100 rows (adjust the batch size as needed)
process_in_batches(hngrams_df, ngrams_df, batch_size=100)

logging.info("All batches processed and hngrams.csv updated.")
