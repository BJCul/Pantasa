import scipy.stats as stats
import pandas as pd

# Define the data
data = {
    "System": ["BERT-Base", "RoBERTa-Base", "RoBERTa-Large", "PANTASA"],
    "Precision": [60.73, 65.95, 67.94, 42.12],
    "Recall": [77.75, 84.69, 84.76, 22.18],
    "F0_5": [63.51, 69.00, 70.75, 35.70],  # Renamed 'F0.5' to 'F0_5'
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Function to perform t-tests for each metric
def t_test_comparison(metric):
    pantasa_values = df[df["System"] == "PANTASA"][metric].values
    results = {}
    for system in df["System"].unique():
        if system != "PANTASA":
            other_values = df[df["System"] == system][metric].values
            t_stat, p_value = stats.ttest_ind(pantasa_values, other_values, equal_var=False)  # Use Welch's t-test
            results[system] = (t_stat, p_value)
    return results

# Perform t-tests for each metric
metrics = ["Precision", "Recall", "F0_5"]
for metric in metrics:
    results = t_test_comparison(metric)
    print(f"\nT-Test Results for {metric}:")
    for system, (t_stat, p_value) in results.items():
        print(f"  PANTASA vs {system}: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
