def remove_blank_lines(input_file):
    with open(input_file, "r+", encoding='utf-8') as f:
        d = f.readlines()
        f.seek(0)
        for i in d:
            if i != "\n":
                f.write(i)
        f.truncate()

def remove_repeating_lines(file_path):
    seen_lines = set()
    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    with open(file_path, 'w', encoding='utf-8') as outfile:
        for line in lines:
            if line not in seen_lines:  # If the line hasn't been seen before
                outfile.write(line)
                seen_lines.add(line)

# Example usage
# input_file = 'database/tokenized.txt'

# remove_blank_lines(input_file)
# remove_repeating_lines(input_file)