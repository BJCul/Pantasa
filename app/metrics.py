import pandas as pd

def compute_metrics(error_free_results_csv, erroneous_results_csv):
    """
    Computes TP, FP, TN, and FN metrics based on the results from the error-free and erroneous datasets.

    Args:
    - error_free_results_csv: Path to the CSV file containing results for error-free sentences.
    - erroneous_results_csv: Path to the CSV file containing results for erroneous sentences.

    Returns:
    - A dictionary containing TP, FP, TN, and FN counts.
    """
    # Initialize counters
    TP = FP = TN = FN = 0

    # Read the results CSV files
    df_error_free = pd.read_csv(error_free_results_csv)
    df_erroneous = pd.read_csv(erroneous_results_csv)

    # Process error-free results
    for idx, row in df_error_free.iterrows():
        original_sentence = row['Original Sentence']
        corrected_sentence = row['Corrected Sentence']
        incorrect_words = row['Incorrect Words']
        spell_suggestions = row['Suggestions']

        # Determine if corrections were made
        corrections_made = (
            pd.notna(corrected_sentence) and original_sentence != corrected_sentence or
            pd.notna(incorrect_words) and incorrect_words not in [None, '[]', '', 'nan'] or
            pd.notna(spell_suggestions) and spell_suggestions not in [None, '{}', '', 'nan']
        )

        if corrections_made:
            # False Positive: Corrections made on an error-free sentence
            FP += 1
        else:
            # True Negative: No corrections made on an error-free sentence
            TN += 1

    # Process erroneous results
    for idx, row in df_erroneous.iterrows():
        original_sentence = row['Original Sentence']
        corrected_sentence = row['Corrected Sentence']
        incorrect_words = row['Incorrect Words']
        spell_suggestions = row['Suggestions']

        # Determine if corrections were made
        corrections_made = (
            pd.notna(corrected_sentence) and original_sentence != corrected_sentence or
            pd.notna(incorrect_words) and incorrect_words not in [None, '[]', '', 'nan'] or
            pd.notna(spell_suggestions) and spell_suggestions not in [None, '{}', '', 'nan']
        )

        if corrections_made:
            # True Positive: Corrections made on an erroneous sentence
            TP += 1
        else:
            # False Negative: No corrections made on an erroneous sentence
            FN += 1

    # Compile the results
    metrics = {
        'True Positive': TP,
        'False Positive': FP,
        'True Negative': TN,
        'False Negative': FN
    }

    # Print the results
    print("Evaluation Metrics:")
    print(metrics)

    return metrics

# Example usage:
if __name__ == "__main__":
    # Paths to your results CSV files
    error_free_results_csv = 'data/processed/error_free_output_result.csv'    # Replace with your actual path
    erroneous_results_csv = 'data/processed/erroneous_output_result.csv'      # Replace with your actual path

    # Compute the metrics
    metrics = compute_metrics(error_free_results_csv, erroneous_results_csv)

    def calculate_additional_metrics(metrics):
        TP = metrics['True Positive']
        FP = metrics['False Positive']
        TN = metrics['True Negative']
        FN = metrics['False Negative']

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        print("\nAdditional Evaluation Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1_score:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")

    calculate_additional_metrics(metrics)
