import pandas as pd

def delete_low_frequency_rows(input_csv, output_csv):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Filter rows where 'frequency' is less than 2
    filtered_df = df[df['Frequency'] >= 10]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"Rows with frequency less than 2 have been removed. Updated CSV saved to {output_csv}")

# Example usage:
input_csv = 'C:\Projects\Pantasa\data\processed\DetailedPOS_N-Gram_Frequency_Counts.csv'
output_csv = 'filtered_hngrams.csv'
delete_low_frequency_rows(input_csv, output_csv)
