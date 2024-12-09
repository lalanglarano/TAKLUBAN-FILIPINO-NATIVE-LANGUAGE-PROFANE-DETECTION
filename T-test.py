import numpy as np
from scipy.stats import ttest_ind

# Confusion matrix totals for TAKLUBAN and BERT
TAKLUBAN = {'TP': 1217, 'TN': 1338, 'FN': 201, 'FP': 259}
BERT = {'TP': 1111, 'TN': 1576, 'FN': 307, 'FP': 21}

testing_size = 3015  # testing size

# Calculate precision, recall, and F-measure for both tools
def calculate_metrics(confusion_matrix):
    TP = confusion_matrix['TP']
    TN = confusion_matrix['TN']
    FN = confusion_matrix['FN']
    FP = confusion_matrix['FP']
    
    # Precision = TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    # Recall = TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    # F-measure = 2 * (Precision * Recall) / (Precision + Recall)
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, f_measure

# Calculate metrics for both TAKLUBAN and BERT
precision_tak, recall_tak, fmeasure_tak = calculate_metrics(TAKLUBAN)
precision_bert, recall_bert, fmeasure_bert = calculate_metrics(BERT)

# Perform T-test on the metrics (precision, recall, F-measure)
metrics = ['Precision', 'Recall', 'F-measure']
results = {}

# Simulate data for the metrics to perform T-test
simulated_data_takluban = {
    'Precision': np.random.normal(precision_tak, 0.02, testing_size),
    'Recall': np.random.normal(recall_tak, 0.02, testing_size),
    'F-measure': np.random.normal(fmeasure_tak, 0.02, testing_size)
}

simulated_data_bert = {
    'Precision': np.random.normal(precision_bert, 0.02, testing_size),
    'Recall': np.random.normal(recall_bert, 0.02, testing_size),
    'F-measure': np.random.normal(fmeasure_bert, 0.02, testing_size)
}

# Perform T-test for each metric
for metric in metrics:
    t_stat, p_value = ttest_ind(simulated_data_takluban[metric], simulated_data_bert[metric])
    results[metric] = {'t_stat': t_stat, 'p_value': p_value}

# Print results with p-value rounded to 5 decimal places
print(f"{'Metric':<12}{'TAKLUBAN':<12}{'BERT':<12}{'p-value':<20}{'Significance':<15}")
for metric in metrics:
    significance = "\t\tNot significant" if results[metric]['p_value'] >= 0.05 else "\t\tSignificant"
    # Printing the metric values and p-values for both tools
    if metric == 'Precision':
        print(f"{metric:<12}{precision_tak:<12.3f}{precision_bert:<12.3f}{results[metric]['p_value']:.5f}{significance}")
    elif metric == 'Recall':
        print(f"{metric:<12}{recall_tak:<12.3f}{recall_bert:<12.3f}{results[metric]['p_value']:.5f}{significance}")
    elif metric == 'F-measure':
        print(f"{metric:<12}{fmeasure_tak:<12.3f}{fmeasure_bert:<12.3f}{results[metric]['p_value']:.5f}{significance}")
