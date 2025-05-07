import json
from together import Together
from sklearn import metrics
import pandas as pd
from tqdm import tqdm

# Load the testing data
with open('../json/testing_data_1.json', 'r') as file:
    test_data_1 = json.load(file)

# Combine test data
test_data = test_data_1

# Initialize the Together client
client = Together(api_key="1538e1a79bb33932ef616714476f3bec56873bc59fee18b8807de5befc80609c")

# Define the prompt
prompt = "验证两段输入文本是否由同一位作者撰写。分析输入文本的写作风格，忽略主题和内容的差异。推理应基于语言特征，例如动词、标点符号、稀有词汇、词缀、幽默、讽刺、打字错误和拼写错误等。输出应遵循以下格式：0 或 1（0表示不同作者，1表示相同作者）。"

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

results_df.to_csv('../csv/zero_shot_results_df_1.csv', index=False)