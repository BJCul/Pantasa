import pandas as pd

# Load the CSV file
file_path = 'data/processed/correct_output_result_d2.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Check for rows where the last three columns are empty
filtered_data = data[
    (data['Corrected Sentence'].isnull() | (data['Corrected Sentence'] == '')) &
    (data['Incorrect Words'].isnull() | (data['Incorrect Words'] == '')) &
    (data['Spell Suggestions'].isnull() | (data['Spell Suggestions'] == ''))
]

# Save the filtered data to a new CSV file
output_path = 'data/processed/correct_output_result_d3.csv'  # Replace with your desired output file path
filtered_data.to_csv(output_path, index=False)

print(f"Filtered lines saved to {output_path}")
