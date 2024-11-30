import csv

def get_latest_pattern_id(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            pattern_ids = [
                int(row['Pattern_ID']) 
                for row in reader 
                if row['Pattern_ID'].isdigit() and len(row['Pattern_ID']) == 6 and row['Pattern_ID'].startswith('0')
            ]
            return max(pattern_ids, default=600000)  # Starting from a default base for safety
    except FileNotFoundError:
        return 600000  # Default starting point if file not found
    
def generate_pattern_id(counter):
    return f"{counter:06d}"

pattern_csv = 'rules/database/POS/6grams.csv'
output_csv = 'rules/database/Generalized/POSTComparison/6grams.csv'

latest_pattern_id = get_latest_pattern_id(pattern_csv)
print(f"Current latest pattern id: {latest_pattern_id}")
pattern_counter = int(str(latest_pattern_id)[1:]) + 1

with open(output_csv, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    updated_rows = []
    
    for row in reader:
        if not (row['Pattern_ID'].isdigit() and len(row['Pattern_ID']) == 6):
            row['Pattern_ID'] = generate_pattern_id(pattern_counter)
            pattern_counter += 1
        elif (row['Pattern_ID'].isdigit() and len(row['Pattern_ID']) == 6 and row['Pattern_ID'].startswith('0')):
            row['Pattern_ID'] = generate_pattern_id(pattern_counter)
            pattern_counter += 1
        updated_rows.append(row)
        print(f"Generated ID {row['Pattern_ID']} for {row['POS_N-Gram']}")

# Save the updated rows to a new file
with open(output_csv, 'w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"All IDs from {output_csv} updated")