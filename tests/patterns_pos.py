import pandas as pd

# Load the CSV file
file_path = 'data/processed/detailed_hngram1.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Debug: Verify DataFrame and column
print(df.head())
print(df.columns)

# Function to extract POS tags with one or more underscores
def extract_unique_pos_tags_with_underscores(pos_column):
    pos_tags = set()
    for pos in pos_column.dropna():
        pos = pos.strip()  # Handle leading/trailing whitespace
        if pos:
            tags = pos.split()
            for tag in tags:
                if '_' in tag:
                    pos_tags.add(tag)
    return pos_tags

# Extract POS tags from the 'Detailed_POS' column
unique_pos_tags = extract_unique_pos_tags_with_underscores(df['Hybrid_N-Gram'])

# Convert the set to a sorted list
unique_pos_tags_list = sorted(unique_pos_tags)

# Debug: Check the extracted tags
print(unique_pos_tags_list)

# Output the results
output_file = 'data/processed/unique_pos_tags_with_underscores.txt'
with open(output_file, 'w') as f:
    for tag in unique_pos_tags_list:
        f.write(f"{tag}\n")

print(f"Unique POS tags with underscores saved to {output_file}.")
