import json
from together import Together
from sklearn import metrics
import pandas as pd
from tqdm import tqdm

# Load the testing data
with open('testing_data_1.json', 'r') as file:
    test_data_1 = json.load(file)

with open('testing_data_2.json', 'r') as file:
    test_data_2 = json.load(file)

# Combine test data
test_data = test_data_2

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
    true_label = item[2]  # 1 for same author, 0 for different authors
    genres_1 = item[0]['genre']
    genres_2 = item[1]['genre']
    mode = 'per-genre' if set(genres_1) & set(genres_2) else 'cross-genre'

    # Make a prediction using the model
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        messages=[{"role": "user", "content": f"{prompt}\nText 1: {text1}\nText 2: {text2}"}]
    )

    # Get the response content
    response_content = response.choices[0].message.content.strip()

    # Find the position of the </think> token
    think_end_pos = response_content.find('</think>')

    # Extract the integer value that follows the </think> token
    if think_end_pos != -1:
        prediction_str = response_content[think_end_pos + len('</think>'):].strip()
        try:
            prediction = int(prediction_str)
        except ValueError:
            prediction = 0  # Assign a default value if conversion fails
    else:
        # Handle the case where </think> is not found
        prediction = 0  # Assign a default value or log an error

    # Append the results
    results.append({'Text1': text1, 'Text2': text2, 'Genre1': genres_1, 'Genre2': genres_2, 'TrueLabel': true_label, 'Prediction': prediction, 'Mode': mode})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate and print the metrics
acc, f1_weighted, f1_micro, f1_macro, recall, precision = evaluate_metrics(results_df['TrueLabel'], results_df['Prediction'])
print(f"Accuracy: {acc}")
print(f"F1 Weighted: {f1_weighted}")
print(f"F1 Micro: {f1_micro}")
print(f"F1 Macro: {f1_macro}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")

# Collect genres from the data
all_genres = set()
for item in test_data:
    all_genres.update(item[0]['genre'])
    all_genres.update(item[1]['genre'])

# Function to calculate metrics per genre
def calculate_metrics_per_genre(results_df, genres):
    metrics_per_genre = {}
    for genre in genres:
        genre_mask = results_df.apply(lambda row: genre in row['Genre1'] or genre in row['Genre2'], axis=1)
        y_true = results_df.loc[genre_mask, 'TrueLabel']
        y_pred = results_df.loc[genre_mask, 'Prediction']
        if not y_true.empty:
            acc = metrics.accuracy_score(y_true, y_pred)
            f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
            f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
            f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
            recall = metrics.recall_score(y_true, y_pred, average='macro')
            precision = metrics.precision_score(y_true, y_pred, average='macro')
            metrics_per_genre[genre] = {
                'Accuracy': acc,
                'F1 Micro': f1_micro,
                'F1 Weighted': f1_weighted,
                'F1 Macro': f1_macro,
                'Recall': recall,
                'Precision': precision
            }
    return metrics_per_genre

# Calculate and print the metrics per genre
metrics_per_genre = calculate_metrics_per_genre(results_df, all_genres)
for genre, metrics in metrics_per_genre.items():
    print(f"Metrics for {genre}: {metrics}")