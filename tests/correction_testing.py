import pandas as pd

def evaluate_correction(results_csv):
    """
    Evaluates correction accuracy by comparing the system's corrected sentences
    to the gold standard from a single CSV file containing:
    Original Sentence, Gold Sentence, Corrected Sentence, Incorrect Words, Spell Suggestions.

    - TP: The model corrected the sentence, and the correction done was correct. nagcorrect at tama
    - FP: The model corrected the sentence, but the correction was wrong or unnecessary. nagcorrect pero mali
    - FN: The model did not correct the sentence, but the data needed correction. Mali tas hindi nagcorrect
    - TN: The model did not correct the sentence, and the data didn’t need correction. tama at hindi nagcorrect

    We compute Precision, Recall, F1, and Accuracy.
    """

    # Read the result CSV
    df_results = pd.read_csv(results_csv)

    # Ensure required columns are present
    required_columns = {'Original Sentence', 'Gold Sentence', 'Corrected Sentence'}
    if not required_columns.issubset(df_results.columns):
        raise ValueError(f"Input CSV must contain the following columns: {required_columns}")

    # Initialize counters
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for idx, row in df_results.iterrows():
        original = row['Original Sentence']
        gold = row['Gold Sentence']
        corrected = row['Corrected Sentence']

        # Determine if the sentence is error-filled (original != gold)
        error_filled = original != gold

        # Evaluate based on the correction status
        if error_filled:  # Sentence contains errors
            if corrected == gold:
                TP += 1  # Correct correction
            elif corrected == original or pd.isna(corrected):
                FN += 1  # Missed correction
            else:
                FP += 1  # Incorrect correction
        else:  # Sentence is error-free
            if corrected == original or pd.isna(corrected):
                TN += 1  # Correctly did not change
            else:
                FP += 1  # Unnecessary correction

    # Total predictions
    total_predictions = TP + FP + TN + FN
    (f"Total predictions: {total_predictions}")

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (1.25 * precision * recall / (0.25 * precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total_predictions if total_predictions > 0 else 0

    # Return results
    return TP, FP, TN, FN, precision, recall, f1_score, accuracy, total_predictions

if __name__ == "__main__":
    # Replace this with the actual path to your results CSV file
    results_csv = 'data/processed/correction_output_result.csv'

    TP, FP, TN, FN, precision, recall, f1_score, accuracy = evaluate_correction(results_csv)

    print("Correction Results:")

    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print("\nCoNLL-Style Correction Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1_score:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
