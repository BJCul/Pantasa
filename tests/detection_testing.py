import pandas as pd

def is_correction_made(row):
    original = row['Original Sentence']
    corrected = row['Corrected Sentence']

    diff_in_sentence = (pd.notna(corrected) and corrected != original)


    return diff_in_sentence

def evaluate_detection(error_free_results_csv, erroneous_results_csv):
    # Load the results
    df_error_free_result = pd.read_csv(error_free_results_csv)
    df_erroneous_result = pd.read_csv(erroneous_results_csv)

    TP = FP = TN = FN = 0

    # Evaluate error-free sentences (no errors)
    for _, row in df_error_free_result.iterrows():
        detected = is_correction_made(row)
        if detected:
            # False Positive - model incorrectly detects an error when there is none
            FP += 1
        else:
            # True Negative - model correctly detects no error
            TN += 1

    # Evaluate erroneous sentences (errors present)
    for _, row in df_erroneous_result.iterrows():
        detected = is_correction_made(row)
        if detected:
            # True Positive - model correctly detects an error
            TP += 1
        else:
            # False Negative - model fails to detect an error when there is one
            FN += 1

    return TP, FP, TN, FN

if __name__ == "__main__":
    # Replace with your actual file paths
    error_free_results_csv = 'data/processed/test_sample_output.csv'
    erroneous_results_csv = 'data/processed/test_sample_output.csv'

    TP, FP, TN, FN = evaluate_detection(error_free_results_csv, erroneous_results_csv)

    print("Detection Results:")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

    # Calculate CoNLL-2014 style metrics: Precision, Recall, F1, and Accuracy
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (1.25 * precision * recall / (0.25 *precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    print("\nCoNLL-Style Detection Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1_score:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
