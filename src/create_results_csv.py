import os
import csv

out_csv = os.path.join('..', 'output', 'results.csv')
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

header = [
    'kernel_type',
    'decision_function_shape',
    'HOG Cell Size',
    'Hu Moments Scaled Up',
    'HOG Disabled',
    'best_C',
    'cv_accuracy',
    'test_accuracy'
]

# Write header (assuming the file doesn't exist yet or has no header)
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
