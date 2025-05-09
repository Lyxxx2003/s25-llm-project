import json
from collections import defaultdict, Counter
import pandas as pd

# === Function to compute genre label distribution ===
def compute_label_distribution(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    genre_label_dist = defaultdict(Counter)
    for s1, s2, label in data:
        genres = set(s1["genre"] + s2["genre"])
        for genre in genres:
            genre_label_dist[genre][label] += 1

    genre_stats = []
    for genre, counts in genre_label_dist.items():
        total = sum(counts.values())
        label_0 = counts[0]
        label_1 = counts[1]
        proportion_0 = label_0 / total if total else 0
        proportion_1 = label_1 / total if total else 0
        genre_stats.append({
            "Genre": genre,
            "Label 0 Count": label_0,
            "Label 1 Count": label_1,
            "Total": total,
            "Label 0 %": round(proportion_0 * 100, 2),
            "Label 1 %": round(proportion_1 * 100, 2)
        })
    return pd.DataFrame(genre_stats).sort_values("Genre")

# === Compare training, test_1, and test_2 ===
paths = {
    "Train": "../json/training_data.json",
    "Test 1": "../json/testing_data_1.json",
    "Test 2": "../json/testing_data_2.json"
}

for name, path in paths.items():
    print(f"\n{name} Distribution:\n")
    df = compute_label_distribution(path)
    print(df.to_string(index=False))
