import pandas as pd
import torch
import json
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the testing data
with open('testing_data_2.json', 'r', encoding='utf-8') as file:
    test_data_2 = json.load(file)

test_data = test_data_2

# Load your fine-tuned model
model_dir = "./chinese_roberta_lyrics_finetuned"  # the directory you saved your model to
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare the test inputs
texts = []
labels = []
genres1 = []
genres2 = []
modes = []

for entry in test_data:
    song1, song2, label = entry
    combined_text = song1['lyrics'] + "\n" + song2['lyrics']
    texts.append(combined_text)
    labels.append(label)
    genres1.append(song1['genre'])
    genres2.append(song2['genre'])
    # Determine if it's "per-genre" or "cross-genre"
    mode = "per-genre" if set(song1['genre']) == set(song2['genre']) else "cross-genre"
    modes.append(mode)

# Batch prediction
batch_size = 16
predictions = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    predictions.extend(preds)

# Create a DataFrame of results
results_df = pd.DataFrame({
    'Text': texts,
    'TrueLabel': labels,
    'Prediction': predictions,
    'Genre1': genres1,
    'Genre2': genres2,
    'Mode': modes
})

# Save results to a CSV (optional)
results_df.to_csv('model_results.csv', index=False)

# Evaluation metrics functions
def evaluate_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    return acc, f1_weighted, f1_micro, f1_macro, recall, precision

# Collect all genres
all_genres = set()
for g_list in genres1 + genres2:
    all_genres.update(g_list)

# Calculate metrics per genre
def calculate_metrics_per_genre(results_df, genres):
    metrics_per_genre = []
    for genre in genres:
        for mode in ['per-genre', 'cross-genre']:
            genre_mask = results_df.apply(
                lambda row: (genre in row['Genre1'] or genre in row['Genre2']) and row['Mode'] == mode, axis=1
            )
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

print("\nMetrics per Genre:")
print(metrics_genre_df.to_string(index=False))

print("\nMetrics per Mode:")
print(metrics_mode_df.to_string(index=False))