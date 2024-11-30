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

detailed_taglist = hierarchical_pos_tags.values()

def tag_type(tag):
    # Check if the tag is a combined rough POS tag (i.e., "NN.*_VB.*")
    if "_" in tag:
        components = tag.split("_")
        # Check if all components are rough POS tags
        if all(component in hierarchical_pos_tags for component in components):
            return "rough POS tag"
        # Check if all components are detailed POS tags
        elif all(any(component in detailed_tags for detailed_tags in hierarchical_pos_tags.values()) for component in components):
            return "detailed POS tag"
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
    logging.debug(f"Detailed POS pattern: {word_search_pattern}")
    
    return rough_pos_search_pattern, detailed_pos_search_pattern, word_search_pattern

# Function to apply rough POS, detailed POS, and word-based filtering
def instance_collector(pattern, ngrams_df, pattern_ngram_size):
    logging.debug(f"Searching n-gram matches for pattern id: {pattern}")
    
    # Step 1: Filter by n-gram size
    size_filtered_df = filter_by_ngram_size(pattern, ngrams_df, pattern_ngram_size)

    # Step 2: Get the three search patterns (rough POS, detailed POS, and words)
    rough_pos_search_pattern, detailed_pos_search_pattern, word_search_pattern = search_pattern_conversion_based_on_tag_type(pattern)
    
    # Step 3: Apply rough POS filtering
    rough_pos_matches = size_filtered_df[size_filtered_df['RoughPOS_N-Gram'].str.contains(rough_pos_search_pattern, regex=True)]
    
    # Step 4: Apply detailed POS filtering on the rough POS matches
    detailed_pos_matches = rough_pos_matches[rough_pos_matches['DetailedPOS_N-Gram'].str.contains(detailed_pos_search_pattern, regex=True)]

    # Step 5: Apply word filtering on thhe detailed POS matches
    word_matches = detailed_pos_matches[detailed_pos_matches['N-Gram'].str.contains(word_search_pattern, regex=True)]
    
    return word_matches

# Define a function to validate the patterns based on your rules
def is_valid_mixed_pos(row):
    logging.debug(f"Validating row with Pattern_ID: {row.get('Pattern_ID', 'Unknown')}")
    
    # Ensure all key columns are non-missing
    if pd.isna(row['RoughPOS_N-Gram']) or pd.isna(row['DetailedPOS_N-Gram']) or pd.isna(row['Comparison_Replacement_Matrix']):
        logging.debug(f"Row rejected due to missing values in key columns: {row}")
        return False

    # Extract the column values
    rough_pos = row['RoughPOS_N-Gram']
    detailed_pos = row['DetailedPOS_N-Gram']
    replacement_matrix = row['Comparison_Replacement_Matrix']

    # Check if RoughPOS and DetailedPOS are non-empty
    if not rough_pos.strip() or not detailed_pos.strip():
        logging.debug(f"Row rejected due to empty RoughPOS or DetailedPOS: {row}")
        return False

    # If all checks pass, the row is valid
    logging.debug(f"Row accepted with mixed POS patterns: {row}")
    return True


# Function to process mixed POS patterns
def process_mixed_pos_patterns(generalized_patterns_df, clusters_df, ngrams_df):
    logging.info("Starting to process mixed POS patterns.")

    # Identify rows with mixed POS sequences and non-empty comparison matrix
    mixed_pos_df = generalized_patterns_df[generalized_patterns_df.apply(is_valid_mixed_pos, axis=1)]
    logging.info(f"Found {len(mixed_pos_df)} valid mixed POS patterns.")

    new_clusters = []

    for index, row in mixed_pos_df.iterrows():
        pattern_id = row['Pattern_ID']
        rough_pos_ngram = row['RoughPOS_N-Gram']
        detailed_pos_ngram = row['DetailedPOS_N-Gram']
        ngram_size = len(rough_pos_ngram.split())  # Assume n-gram size from rough POS sequence

        logging.debug(f"Processing Pattern_ID {pattern_id} with RoughPOS {rough_pos_ngram}")

        # Collect matching n-gram IDs using instance_collector
        matched_instances = instance_collector(
            rough_pos_ngram, 
            ngrams_df, 
            ngram_size
        )
        id_array = matched_instances['N-Gram_ID'].tolist()
        frequency = len(id_array)

        logging.debug(f"Matched {frequency} instances for Pattern_ID {pattern_id}")

        if len(id_array) != 0:
            # Construct new cluster entry
            new_cluster = {
                "Pattern_ID": pattern_id,
                "RoughPOS_N-Gram": rough_pos_ngram,
                "DetailedPOS_N-Gram": detailed_pos_ngram,
                "Frequency": frequency,
                "ID_Array": str(id_array)  # Convert list to string for storage
            }
            new_clusters.append(new_cluster)
        else:
            # Remove the row from the generalized_patterns_df if no matches are found
            generalized_patterns_df.drop(index, inplace=True)

    # Append new clusters to the clusters dataframe
    new_clusters_df = pd.DataFrame(new_clusters)
    updated_clusters_df = pd.concat([clusters_df, new_clusters_df], ignore_index=True)

    logging.info("Mixed POS patterns processing complete.")
    return updated_clusters_df



generalized_patterns_file = "rules/database/Generalized/POSTComparison/6grams.csv"
clusters_file = "rules/database/POS/6grams.csv"
ngrams_file = "rules/database/ngram.csv"

logging.info("Loading input files.")
generalized_patterns_df = pd.read_csv(generalized_patterns_file)
clusters_df = pd.read_csv(clusters_file)
ngrams_df = pd.read_csv(ngrams_file)

# Process the mixed POS patterns and update the clusters file
logging.info("Processing mixed POS patterns.")
updated_clusters_df = process_mixed_pos_patterns(generalized_patterns_df, clusters_df, ngrams_df)

# Save the updated clusters file
output_file = "rules/database/POS/updated_clusters.csv"
logging.info(f"Saving updated clusters to {output_file}.")
updated_clusters_df.to_csv(output_file, index=False)

logging.info("Clusters file updated successfully.")