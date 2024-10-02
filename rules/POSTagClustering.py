import csv
from collections import defaultdict

def generate_pattern_id(ngram_size, counter):
    return f"{ngram_size}{counter:05d}"

def get_max_pattern_id(output_file):
    max_pattern_id = 0
    try:
        with open(output_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pattern_id = int(row['Pattern_ID'][1:])  # Skipping the first character
                if pattern_id > max_pattern_id:
                    max_pattern_id = pattern_id
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error reading max pattern ID: {e}")
    return max_pattern_id

def process_clustered_ngrams(input_file, output_file, ngram_size):
    rough_pos_patterns = defaultdict(list)
    detailed_pos_patterns = defaultdict(list)
    max_pattern_id = get_max_pattern_id(output_file)
    
    # Read the clustered n-grams from the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            
            for row in reader:
                ngram_id = row['N-Gram_ID']
                rough_pos = row['RoughPOS_N-Gram']
                detailed_pos = row['DetailedPOS_N-Gram']
                
                if rough_pos:
                    rough_pos_patterns[rough_pos].append(ngram_id)
                if detailed_pos:
                    detailed_pos_patterns[detailed_pos].append(ngram_id)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Load existing patterns from the output file to avoid duplicates
    existing_patterns = {}
    try:
        with open(output_file, 'r', encoding='utf-8') as out_file:
            reader = csv.DictReader(out_file)
            for row in reader:
                pattern_id = row['Pattern_ID']
                rough_pos = row['RoughPOS_N-Gram']
                detailed_pos = row['DetailedPOS_N-Gram']
                frequency = int(row['Frequency'])
                id_array = row['ID_Array'].strip('[]').replace('\'', '').split(', ')
                
                existing_patterns[(rough_pos, detailed_pos)] = {
                    'Pattern_ID': pattern_id,
                    'Frequency': frequency,
                    'ID_Array': id_array
                }
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error reading output file: {e}")
    
    # Write the results to the output CSV file
    try:
        with open(output_file, 'a', newline='', encoding='utf-8') as out_file:
            fieldnames = ['Pattern_ID', 'RoughPOS_N-Gram', 'DetailedPOS_N-Gram', 'Frequency', 'ID_Array']
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            
            if max_pattern_id == 0:
                writer.writeheader()
            
            # Write rough POS patterns
            for rough_pos, ids in rough_pos_patterns.items():
                if len(ids) > 1:  # Only include patterns with two or more occurrences
                    existing = existing_patterns.get((rough_pos, ''))
                    if existing:
                        # Update existing pattern
                        existing['Frequency'] += len(ids)
                        existing['ID_Array'].extend(ids)
                    else:
                        # New pattern
                        pattern_id = generate_pattern_id(ngram_size, max_pattern_id + 1)
                        max_pattern_id += 1
                        writer.writerow({
                            'Pattern_ID': pattern_id,
                            'RoughPOS_N-Gram': rough_pos,
                            'DetailedPOS_N-Gram': '',
                            'Frequency': len(ids),
                            'ID_Array': ids
                        })
            
            # Write detailed POS patterns
            for detailed_pos, ids in detailed_pos_patterns.items():
                if len(ids) > 1:  # Only include patterns with two or more occurrences
                    existing = existing_patterns.get(('', detailed_pos))
                    if existing:
                        # Update existing pattern
                        existing['Frequency'] += len(ids)
                        existing['ID_Array'].extend(ids)
                    else:
                        # New pattern
                        pattern_id = generate_pattern_id(ngram_size, max_pattern_id + 1)
                        max_pattern_id += 1
                        writer.writerow({
                            'Pattern_ID': pattern_id,
                            'RoughPOS_N-Gram': '',
                            'DetailedPOS_N-Gram': detailed_pos,
                            'Frequency': len(ids),
                            'ID_Array': ids
                        })
            
            # Write updated patterns
            for key, value in existing_patterns.items():
                if value['Frequency'] > 1:  # Ensure patterns have at least two occurrences
                    writer.writerow({
                        'Pattern_ID': value['Pattern_ID'],
                        'RoughPOS_N-Gram': key[0],
                        'DetailedPOS_N-Gram': key[1],
                        'Frequency': value['Frequency'],
                        'ID_Array': value['ID_Array']
                    })
                
    except Exception as e:
        print(f"Error writing output file: {e}")

# Example usage
for n in range(2, 8):
    input_csv = f'database/GramSize/{n}grams.csv'
    output_csv = f'database/POS/{n}grams.csv'
    process_clustered_ngrams(input_csv, output_csv, n)
