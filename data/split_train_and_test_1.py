import json
import random
from collections import defaultdict

with open("../json/songs_data_filtered_Chinese.json", "r") as file:
    all_data = json.load(file)

all_data = list(all_data.values())


# ----------------------------
# Helper: Category + Mode
# ----------------------------
def get_pair_category_and_mode(s1, s2):
    same_author = (s1["lyricist(s)"] == s2["lyricist(s)"])
    same_genre = bool(set(s1["genre"]) & set(s2["genre"]))

    if same_author and same_genre:
        category = 1
    elif (not same_author) and same_genre:
        category = 2
    elif (not same_author) and (not same_genre):
        category = 3
    else:
        category = 4

    mode = "per-genre" if same_genre else "cross-genre"
    label = 1 if same_author else 0
    return category, mode, label


# ----------------------------
# Build ALL UNIQUE PAIRS
# ----------------------------
used_pairs = set()
all_pairs = []

def pair_key(a, b):
    return tuple(sorted([a["lyrics"], b["lyrics"]]))

for i in range(len(all_data)):
    for j in range(i + 1, len(all_data)):
        s1, s2 = all_data[i], all_data[j]
        key = pair_key(s1, s2)
        if key in used_pairs:
            continue
        used_pairs.add(key)

        cat, mode, label = get_pair_category_and_mode(s1, s2)

        all_pairs.append({
            "song1": s1,
            "song2": s2,
            "category": cat,
            "label": label,
            "mode": mode
        })


# ----------------------------
# Balancing
# ----------------------------
def balance_categories(data):
    cats = {1: [], 2: [], 3: [], 4: []}
    for item in data:
        cats[item["category"]].append(item)

    c1, c2, c3, c4 = map(len, (cats[1], cats[2], cats[3], cats[4]))

    scale = min(c1, c2, c3, c4)

    result = []
    result.extend(cats[1][:scale])
    result.extend(cats[2][:scale])
    result.extend(cats[3][:scale])
    result.extend(cats[4][:scale])

    random.shuffle(result)
    return result

def genre_smoothing(data, M_genre=3):
    genre_buckets = defaultdict(list)
    cross_bucket = []

    for item in data:
        g1 = item["song1"]["genre"]
        g2 = item["song2"]["genre"]
        shared = set(g1) & set(g2)

        if shared:
            genre = sorted(shared)[0]
            genre_buckets[genre].append(item)
        else:
            cross_bucket.append(item)

    genre_sizes = [(g, len(items)) for g, items in genre_buckets.items()]
    genre_sizes.sort(key=lambda x: x[1])

    if len(genre_sizes) <= 2:
        return data

    base = genre_sizes[1][1]
    cap = M_genre * base

    new_genre_buckets = {}

    for g, size in genre_sizes:
        items = genre_buckets[g]
        if size <= base:
            new_genre_buckets[g] = items  # keep smallest genres fully
        else:
            random.shuffle(items)
            new_genre_buckets[g] = items[:cap]

    result = []
    for g, items in new_genre_buckets.items():
        result.extend(items)

    result.extend(cross_bucket)
    random.shuffle(result)
    return result


all_pairs = genre_smoothing(all_pairs, M_genre=1)
all_pairs = balance_categories(all_pairs)


# ----------------------------
# SPLIT
# ----------------------------
# Key = (category, mode)
strata = defaultdict(list)
for item in all_pairs:
    key = (item["category"], item["mode"])
    strata[key].append(item)

train_data = []
test1_data = []

for key, bucket in strata.items():
    random.shuffle(bucket)
    n = len(bucket)
    split_idx = int(0.8 * n)

    train_data.extend(bucket[:split_idx])
    test1_data.extend(bucket[split_idx:])

random.shuffle(train_data)
random.shuffle(test1_data)

with open("../json/training_data.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("../json/testing_data_1.json", "w") as f:
    json.dump(test1_data, f, ensure_ascii=False, indent=2)
