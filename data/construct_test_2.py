import json
import random
from collections import defaultdict

with open("../json/test2_data.json", "r") as file:
    all_data = json.load(file)

all_data = list(all_data.values())

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


test2_data = []
used = set()

def pair_key(a, b):
    return tuple(sorted([a["lyrics"], b["lyrics"]]))

for i in range(len(all_data)):
    for j in range(i + 1, len(all_data)):
        s1, s2 = all_data[i], all_data[j]
        key = pair_key(s1, s2)
        if key in used:
            continue
        used.add(key)

        cat, mode, label = get_pair_category_and_mode(s1, s2)
        test2_data.append({
            "song1": s1, "song2": s2,
            "category": cat, "label": label, "mode": mode
        })

# smoothing multiplier
def balance_categories(data, M=3):
    cats = {1: [], 2: [], 3: [], 4: []}
    for item in data:
        cats[item["category"]].append(item)

    c1, c2, c3, c4 = map(len, (cats[1], cats[2], cats[3], cats[4]))

    scale = max(c1, c4)
    # cap2 = M * scale
    # cap3 = M * scale

    # t2 = min(c2, cap2)
    # t3 = min(c3, cap3)

    random.shuffle(cats[2])
    random.shuffle(cats[3])

    result = []
    result.extend(cats[1][:scale])
    result.extend(cats[4][:scale])
    result.extend(cats[2][:scale])
    result.extend(cats[3][:scale])

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
            # keep smallest & second smallest fully
            new_genre_buckets[g] = items
        else:
            # cap bigger genres
            random.shuffle(items)
            new_genre_buckets[g] = items[:cap]

    result = []
    for g, items in new_genre_buckets.items():
        result.extend(items)

    result.extend(cross_bucket)

    random.shuffle(result)
    return result

test2_data = genre_smoothing(test2_data, M_genre=3)
test2_data = balance_categories(test2_data, M=3)


with open("../json/testing_data_2.json", "w") as f:
    json.dump(test2_data, f, ensure_ascii=False, indent=2)