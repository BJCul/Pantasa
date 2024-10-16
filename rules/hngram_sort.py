import pandas as pd

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
    if tag in hierarchical_pos_tags:
        return "rough POS tag"
    
    # Check if the tag is a detailed POS tag (found in the values of the dictionary)
    for rough_tag, detailed_tags in hierarchical_pos_tags.items():
        if tag in detailed_tags:
            return "detailed POS tag"
    
# Function to return 3 patterns based on rough POS tags, detailed POS tags, and words
def classify_ngram_pos(pattern):
    pattern_parts = pattern.split()
    
    # Separate patterns for rough POS, detailed POS, and words
    rough_pos_pattern = []
    detailed_pos_pattern = []

    for part in pattern_parts:
        tag_category = tag_type(part)
        
        # Rough POS Pattern
        if tag_category == "rough POS tag":
            rough_pos_pattern.append(part)  # Keep rough POS tag
            detailed_pos_pattern.append(r'.*')  # Replace detailed POS with wildcard
        # Detailed POS Pattern
        elif tag_category == "detailed POS tag":
            rough_pos_pattern.append(r'.*')  # Replace rough POS with wildcard
            detailed_pos_pattern.append(part)  # Keep detailed POS tag
    
    # If Rough POS is filled only with ".*", it's considered Only Detailed POS
    if all(tag == '.*' for tag in rough_pos_pattern):
        return 'Only Detailed POS'
    # If Detailed POS is filled only with ".*", it's considered Only Rough POS
    elif all(tag == '.*' for tag in detailed_pos_pattern):
        return 'Only Rough POS'
    # Otherwise, it's Mixed POS
    else:
        return 'Mixed POS'
        
hngrams_df = pd.read_csv('rules/database/hngrams.csv')

# Apply the corrected function to classify n-grams
hngrams_df['POS_Type'] = hngrams_df['Hybrid_N-Gram'].apply(classify_ngram_pos)

# Separate into three categories
only_detailed_pos_df = hngrams_df[hngrams_df['POS_Type'] == 'Only Detailed POS']
only_rough_pos_df = hngrams_df[hngrams_df['POS_Type'] == 'Only Rough POS']
mixed_pos_df = hngrams_df[hngrams_df['POS_Type'] == 'Mixed POS']

# Export the categorized n-grams to CSV files
only_detailed_pos_df.to_csv('rules/database/detailed_hngram.csv', index=False)
only_rough_pos_df.to_csv('rules/database/rough_hngram.csv', index=False)
mixed_pos_df.to_csv('rules/database/mixed_hngram.csv', index=False)
