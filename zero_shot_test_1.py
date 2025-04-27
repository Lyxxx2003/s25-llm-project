import json
from together import Together
from sklearn import metrics
import pandas as pd
from tqdm import tqdm

# Load the testing data
with open('testing_data_1.json', 'r') as file:
    test_data_1 = json.load(file)

# Combine test data
test_data = test_data_1

# Initialize the Together client
client = Together(api_key="")

# Define the prompt
prompt = "验证两段输入文本是否由同一位作者撰写。分析输入文本的写作风格，忽略主题和内容的差异。推理应基于语言特征，例如动词、标点符号、稀有词汇、词缀、幽默、讽刺、打字错误和拼写错误等。输出应遵循以下格式：0 或 1（0表示不同作者，1表示相同作者）。"

# Function to evaluate metrics
def evaluate_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    return acc, f1_weighted, f1_micro, f1_macro, recall, precision

# Placeholder for predictions, true labels, and genres
results = []

# Iterate over the data with a progress bar
for item in tqdm(test_data, desc="Processing data"):
    text1 = item[0]['lyrics']
    text2 = item[1]['lyrics']
    true_label = item[2]
    genres_1 = item[0]['genre']
    genres_2 = item[1]['genre']
    mode = 'per-genre' if set(genres_1) & set(genres_2) else 'cross-genre'

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        messages=[{"role": "user", "content": f"{prompt}\nText 1: {text1}\nText 2: {text2}"}]
    )

    response_content = response.choices[0].message.content.strip()
    think_end_pos = response_content.find('</think>')

    if think_end_pos != -1:
        prediction_str = response_content[think_end_pos + len('</think>'):].strip()
        try:
            prediction = int(prediction_str)
        except ValueError:
            prediction = 0
    else:
        prediction = -1

    results.append({'Text1': text1, 'Text2': text2, 'Genre1': genres_1, 'Genre2': genres_2, 'TrueLabel': true_label, 'Prediction': prediction, 'Mode': mode})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

results_df.to_csv('results_df_1.csv', index=False)

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

print("\nMetrics per Genre:")
print(metrics_genre_df.to_string(index=False))

print("\nMetrics per Mode:")
print(metrics_mode_df.to_string(index=False))