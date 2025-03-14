import pandas as pd
import chardet

def is_correction_made(row):
    original = row['Original Sentence']
    corrected = row['Corrected Sentence']
    incorrect_words = row['Incorrect Words']
    spell_suggestions = row['Spell Suggestions']

    diff_in_sentence = (pd.notna(corrected) and corrected != original)

    return diff_in_sentence

def evaluate_detection(error_free_results_csv, erroneous_results_csv, output_error_free_csv, output_erroneous_csv):
    # Load the results
    with open(error_free_results_csv, "rb") as f:
        result = chardet.detect(f.read())
        print(result)  # This will show the encoding

    df_erroneous_result = pd.read_csv(erroneous_results_csv, encoding=result['encoding'])
    df_error_free_result = pd.read_csv(error_free_results_csv, encoding=result['encoding'])

    TP = FP = TN = FN = 0

    # Evaluate error-free sentences (no errors)
    results = []
    for _, row in df_error_free_result.iterrows():
        detected = is_correction_made(row)
        if detected:
            # False Negative - model detects grammar errors but there are no actual grammar errors in the data
            FN += 1
            results.append("FN")
        else:
            # True Positive - model detects no grammar errors and data has no actual grammar errors
            TP += 1
            results.append("TP")
    df_error_free_result.insert(0, "Detection Result", results)

    # Evaluate erroneous sentences (errors present)
    results = []
    for _, row in df_erroneous_result.iterrows():
        detected = is_correction_made(row)
        if detected:
            # True Negative - model detects grammar errors and there are actual grammar errors in the data
            TP += 1
            results.append("TN")
        else:
            # False Positive - model detects no grammar errors but the data actually has grammar errors
            FP += 1
            results.append("FP")
    df_erroneous_result.insert(0, "Detection Result", results)

    # Save updated CSVs with the new column
    df_error_free_result.to_csv(output_error_free_csv, index=False)
    df_erroneous_result.to_csv(output_erroneous_csv, index=False)

    return TP, FP, TN, FN

if __name__ == "__main__":
    # Replace with your actual file paths
    error_free_results_csv = r"data/processed/correct_output_result_d3_1.csv"
    erroneous_results_csv = r"data/processed/incorrect_output_result_t3_1.csv"
    output_error_free_csv = r"data/processed/raw_result/error_free_output_result_with_labels2.csv"
    output_erroneous_csv = r"data/processed/raw_result/erroneous_output_result_with_labels2.csv"

    TP, FP, TN, FN = evaluate_detection(error_free_results_csv, erroneous_results_csv, output_error_free_csv, output_erroneous_csv)

    print("Detection Results:")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

    # Calculate CoNLL-2014 style metrics: Precision, Recall, F1, and Accuracy
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    print("\nCoNLL-Style Detection Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1_score:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")