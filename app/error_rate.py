import pandas as pd

def compute_error_rate_accuracy(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_columns = ['Original Sentence', 'Gold Sentence', 'Corrected Sentence', 'Incorrect Words', 'Suggestions']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")

    total_predictions = len(df)
    false_positives = 0
    false_negatives = 0

    for idx, row in df.iterrows():
        original = row['Original Sentence']
        gold = row['Gold Sentence']
        corrected = row['Corrected Sentence']
        incorrect_words = str(row['Incorrect Words']) if pd.notna(row['Incorrect Words']) else ""
        suggestions = str(row['Suggestions']) if pd.notna(row['Suggestions']) else ""

        # Determine if the sentence is actually erroneous
        actually_erroneous = (original != gold)

        # Define detected_error based on system output
        # Detected error if system changed something or indicated corrections/suggestions
        detected_error = (
            (corrected != original)
        )

        # Compute FP (False Positive for Detection)
        # FP: Actually erroneous but system did NOT detect error
        if actually_erroneous and not detected_error:
            false_positives += 1

        # Compute FN (False Negative for Correction)
        # FN: Actually erroneous, system detected error, but corrected != gold
        if actually_erroneous and detected_error and (corrected != gold):
            false_negatives += 1

    # Compute Error Rate and Accuracy
    if total_predictions > 0:
        error_rate = (false_positives + false_negatives) / total_predictions
    else:
        error_rate = 0.0  # If no predictions, no error

    accuracy = 1 - error_rate

    # Print or return the results
    print("Performance Results:")
    print(f"Total Predictions: {total_predictions}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Error Rate: {error_rate:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Optionally return results as a dictionary
    return {
        'Total Predictions': total_predictions,
        'False Positives': false_positives,
        'False Negatives': false_negatives,
        'Error Rate': error_rate,
        'Accuracy': accuracy
    }

if __name__ == "__main__":
    csv_path = 'data/processed/mixed.csv'
    compute_error_rate_accuracy(csv_path)
