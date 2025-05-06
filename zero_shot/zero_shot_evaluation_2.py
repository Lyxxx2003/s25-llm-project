import json
import pandas as pd
from sklearn import metrics

# Load the testing data
with open('./json/testing_data_2.json', 'r') as file:
    test_data_2 = json.load(file)

# Combine test data
test_data = test_data_2

results_df = pd.read_csv('./csv/results_df_2.csv')

def evaluate_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    return acc, f1_weighted, f1_micro, f1_macro, recall, precision

# Collect genres
all_genres = set()
for item in test_data:
    all_genres.update(item[0]['genre'])
    all_genres.update(item[1]['genre'])

# Calculate metrics per genre
def calculate_metrics_per_genre(results_df, genres):
    metrics_per_genre = []
    for genre in genres:
        for mode in ['per-genre', 'cross-genre']:
            genre_mask = results_df.apply(lambda row: (genre in row['Genre1'] or genre in row['Genre2']) and row['Mode'] == mode, axis=1)
            if genre_mask.any():
                subset = results_df[genre_mask]
                acc, f1_weighted, f1_micro, f1_macro, recall, precision = evaluate_metrics(subset['TrueLabel'], subset['Prediction'])
                metrics_per_genre.append({
                    'Genre': genre,
                    'Mode': mode,
                    'Accuracy': acc,
                    'F1 Micro': f1_micro,
                    'F1 Weighted': f1_weighted,
                    'F1 Macro': f1_macro,
                    'Recall': recall,
                    'Precision': precision
                })
    return pd.DataFrame(metrics_per_genre)

# Calculate metrics per mode
def calculate_metrics_per_mode(results_df):
    metrics_per_mode = []
    for mode in results_df['Mode'].unique():
        subset = results_df[results_df['Mode'] == mode]
        acc, f1_weighted, f1_micro, f1_macro, recall, precision = evaluate_metrics(subset['TrueLabel'], subset['Prediction'])
        metrics_per_mode.append({
            'Mode': mode,
            'Accuracy': acc,
            'F1 Micro': f1_micro,
            'F1 Weighted': f1_weighted,
            'F1 Macro': f1_macro,
            'Recall': recall,
            'Precision': precision
        })
    return pd.DataFrame(metrics_per_mode)

# Generate and print tables
metrics_genre_df = calculate_metrics_per_genre(results_df, all_genres)
metrics_mode_df = calculate_metrics_per_mode(results_df)

metrics_genre_df.to_csv("./csv/zero_shot_evaluation_2_metrics_genre.csv", index=False)
metrics_mode_df.to_csv("./csv/zero_shot_evaluation_2_metrics_mode.csv", index=False)

print("\nMetrics per Genre:")
print(metrics_genre_df.to_string(index=False))

print("\nMetrics per Mode:")
print(metrics_mode_df.to_string(index=False))