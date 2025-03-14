import pandas as pd
import chardet
import ast

def safe_eval(val):
    """Safely evaluates a string containing a list/dict and returns the actual object."""
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, (list, dict)) else None
    except (SyntaxError, ValueError):
        return None

def evaluate_correction(erroneous_dataset_csv, erroneous_results_csv, output_csv):
    """
    Evaluates correction accuracy by comparing the system's corrected sentences
    to the gold standard and labels each sentence as TP or FN.

    - TP: The corrected sentence matches the gold sentence exactly.
    - FN: The corrected sentence does not match the gold sentence.
    """

    with open(erroneous_dataset_csv, "rb") as f:
        result = chardet.detect(f.read())
        print(result)  # This will show the encoding

    df_erroneous = pd.read_csv(erroneous_dataset_csv, encoding=result['encoding'])
    df_erroneous_result = pd.read_csv(erroneous_results_csv, encoding=result['encoding'])


    # Ensure that the number of rows match (assuming 1:1 correspondence)
    if len(df_erroneous) != len(df_erroneous_result):
        raise ValueError("The number of sentences in the erroneous dataset and the results file do not match.")

    total_erroneous = len(df_erroneous)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    detection_results = []
    correction_results = []

    for idx in range(total_erroneous):
        gold = df_erroneous.loc[idx, 'Gold Sentence']
        system_corrected = df_erroneous_result.loc[idx, 'Corrected Sentence']
        original = df_erroneous.loc[idx, 'Original Sentence']
        errors = safe_eval(df_erroneous_result.loc[idx, 'Incorrect Words']) or []
        suggestions = safe_eval(df_erroneous_result.loc[idx, 'Spell Suggestions']) or {}


        if original.strip() == gold.strip() and (not errors or not suggestions):
            TN += 1
            detection_results.append("TN")
            correction_results.append("TN")
        elif original.strip() == gold.strip() and (errors or suggestions):
            detection_results.append("FP")
            correction_results.append("FP")
        elif original.strip() != gold.strip() and system_corrected.strip() == original.strip():
            FN +=1
            detection_results.append("FN")
            correction_results.append("FN")
        elif original.strip() != gold.strip() and system_corrected.strip() == gold.strip():
            TP += 1
            detection_results.append("TP")
            correction_results.append("TP")
        elif original.strip() != gold.strip() and system_corrected.strip() != gold.strip():
            FN += 1
            detection_results.append("TP")
            correction_results.append("FN")
        


    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # Add detection results column
    df_erroneous_result["Detection Result"] = detection_results
    df_erroneous_result["Correction Result"] = correction_results
    df_erroneous_result.to_csv(output_csv, index=False)

    return TP, FP, TN, FN, precision, recall, f1_score, accuracy

if __name__ == "__main__":
    # Replace these with the actual paths to your dataset and results files
    erroneous_dataset_csv = r"data\processed\incorrect_output_result_t5_1.csv"
    erroneous_results_csv = r"data\processed\incorrect_output_result_t5_1.csv"
    output_csv = r"data\processed\raw_result\correction_evaluation_result33.csv"

    TP, FP, TN, FN, precision, recall, f1_score, accuracy = evaluate_correction(erroneous_dataset_csv, erroneous_results_csv, output_csv)

    print("Correction Results:")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print("\nCoNLL-Style Correction Metrics (Adapted):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1_score:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")