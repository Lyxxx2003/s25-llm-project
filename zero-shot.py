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
test_data = test_data_1[:50] + test_data_2[:50]

# Initialize the Together client
client = Together(api_key="YOUR_API_KEY")

# Define the prompt
prompt = "验证两段输入文本是否由同一位作者撰写。分析输入文本的写作风格，忽略主题和内容的差异。推理应基于语言特征，例如动词、标点符号、稀有词汇、词缀、幽默、讽刺、打字错误和拼写错误等。输出应遵循以下格式：0 或 1（0表示不同作者，1表示相同作者）。"

# Function to evaluate metrics
def evaluate_metrics(y_true, y_pred, genres):
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    f1_per_genre = {genre: metrics.f1_score(y_true, y_pred, labels=[genre], average='micro') for genre in set(genres)}
    return acc, f1_micro, f1_macro, recall, precision, f1_per_genre

# Placeholder for predictions, true labels, and genres
results = []

# Iterate over the data with a progress bar
for item in tqdm(test_data, desc="Processing data"):
    text1 = item[0]['lyrics']
    text2 = item[1]['lyrics']
    true_label = item[2]  # 1 for same author, 0 for different authors
    genres_1 = set(item[0]['genre'])
    genres_2 = set(item[1]['genre'])
    mode = 'per-genre' if genres_1 & genres_2 else 'cross-genre'

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
        prediction = int(prediction_str)
    else:
        # Handle the case where </think> is not found
        prediction = None  # or some default value or error handling

    # Append the results
    results.append({'Text1': text1, 'Text2': text2, 'TrueLabel': true_label, 'Prediction': prediction, 'Mode': mode})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate and print the metrics
acc, f1_micro, f1_macro, recall, precision, f1_per_genre = evaluate_metrics(results_df['TrueLabel'], results_df['Prediction'], results_df['Mode'])
print(f"Accuracy: {acc}")
print(f"F1 Micro: {f1_micro}")
print(f"F1 Macro: {f1_macro}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Per Genre: {f1_per_genre}")
