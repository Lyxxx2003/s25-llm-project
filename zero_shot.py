# zero-shot
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn import metrics

# load from hf instead
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print("Model loaded")

def test_results(filename):

    with open(filename, 'r') as file:
        test_data = json.load(file)

    prompt = "验证两段输入文本是否由同一位作者撰写。分析输入文本的写作风格，忽略主题和内容的差异。推理应基于语言特征，例如动词、标点符号、稀有词汇、词缀、幽默、讽刺、打字错误和拼写错误等。输出应遵循以下格式：0 或 1（0表示不同作者，1表示相同作者）。"

    results = []

    for item in tqdm(test_data, desc="Processing data"):
        text1 = item["song1"]['lyrics']
        text2 = item["song2"]['lyrics']
        true_label = float(item["label"])
        genres_1 = item["song1"]['genre']
        genres_2 = item["song2"]['genre']
        mode = 'per-genre' if set(genres_1) & set(genres_2) else 'cross-genre'

        user_input = f"{prompt}\nText 1: {text1}\nText 2: {text2}"

        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        prediction = 0
        for ch in reversed(response):
            if ch in ("0", "1"):
                prediction = int(ch)
                break
        results.append({'Text1': text1, 'Text2': text2, 'Genre1': genres_1, 'Genre2': genres_2, 'TrueLabel': true_label, 'Prediction': prediction, 'Mode': mode})


    results_df = pd.DataFrame(results)

    def evaluate_metrics(y_true, y_pred):
        acc = metrics.accuracy_score(y_true, y_pred)
        f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
        f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
        f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
        recall = metrics.recall_score(y_true, y_pred, average='macro')
        precision = metrics.precision_score(y_true, y_pred, average='macro')
        return acc, f1_weighted, f1_micro, f1_macro, recall, precision


    all_genres = set()
    for item in test_data:
        if isinstance(item, dict):
            all_genres.update(item['song1']['genre'])
            all_genres.update(item['song2']['genre'])
        else:
            all_genres.update(item[0]['genre'])
            all_genres.update(item[1]['genre'])

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

    metrics_genre_df = calculate_metrics_per_genre(results_df, all_genres)
    metrics_mode_df = calculate_metrics_per_mode(results_df)

    print("\nMetrics per Genre:")
    print(metrics_genre_df.to_string(index=False))

    print("\nMetrics per Mode:")
    print(metrics_mode_df.to_string(index=False))

if __name__ == "__main__":
    test_results('./json/testing_data_1.json')
    test_results('./json/testing_data_2.json')