import pandas as pd

def evaluate_correction(erroneous_dataset_csv, erroneous_results_csv):
    """
    Evaluates correction accuracy by comparing the system's corrected sentences
    to the gold standard.

    - TP: The corrected sentence matches the gold sentence exactly.
    - FN: The corrected sentence does not match the gold sentence.
    - FP, TN are not used here since we're only dealing with originally erroneous sentences.

    We then compute Precision, Recall, F1, and Accuracy in a simplified manner.
    """

    df_erroneous = pd.read_csv(erroneous_dataset_csv)
    df_erroneous_result = pd.read_csv(erroneous_results_csv)

    # Ensure that the number of rows match (assuming 1:1 correspondence)
    if len(df_erroneous) != len(df_erroneous_result):
        raise ValueError("The number of sentences in the erroneous dataset and the results file do not match.")

    total_erroneous = len(df_erroneous)
    TP = 0

    for idx in range(total_erroneous):
        gold = df_erroneous.loc[idx, 'Gold Sentence']
        system_corrected = df_erroneous_result.loc[idx, 'Corrected Sentence']

        if pd.notna(system_corrected) and system_corrected == gold:
            TP += 1
        # Otherwise, FN implicitly
    FN = total_erroneous - TP
    FP = 0
    TN = 0

    # Compute metrics
    # Precision: TP / (TP + FP) → If we consider only erroneous sentences (no FP here) it's just TP/TP = 1 if TP>0
    # Recall: TP / (TP + FN)
    # F1: 2 * (Precision * Recall) / (Precision + Recall)
    # Accuracy: (TP + TN) / (TP + TN + FP + FN) but TN and FP =0 here simplifies to TP/(TP+FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return TP, FP, TN, FN, precision, recall, f1_score, accuracy

if __name__ == "__main__":
    # Replace these with the actual paths to your dataset and results files
    erroneous_dataset_csv = 'data/processed/erroneous_gold.csv'
    erroneous_results_csv = 'data/processed/correction_erroneous_output_result.csv'

    TP, FP, TN, FN, precision, recall, f1_score, accuracy = evaluate_correction(erroneous_dataset_csv, erroneous_results_csv)

    print("Correction Results:")
    print(f"TP: {TP}, FN: {FN}")
    print("\nCoNLL-Style Correction Metrics (Adapted):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1_score:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
