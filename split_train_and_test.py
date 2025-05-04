import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# Load font that supports Chinese
chinese_font = fm.FontProperties(fname="SimHei.ttf")

# -------------------- Load and Preprocess --------------------

with open("songs_data_filtered_Chinese.json", "r") as file:
    songs_data = json.load(file)

# Group songs by lyricist and genre
lyricist_songs = defaultdict(lambda: defaultdict(list))
for _, song_info in songs_data.items():
    lyricist = song_info["lyricist(s)"]
    for genre in song_info["genre"]:
        lyricist_songs[lyricist][genre].append(song_info)

# -------------------- Pair Generation --------------------

def create_hard_positives():
    positives = []
    for lyricist, genres in lyricist_songs.items():
        if len(genres) > 1:
            for g1 in genres:
                for g2 in genres:
                    if g1 != g2 and genres[g1] and genres[g2]:
                        positives.append((random.choice(genres[g1]), random.choice(genres[g2]), 1))
    return positives

def create_hard_negatives():
    negatives = []
    all_genres = set(g for genres in lyricist_songs.values() for g in genres)
    for genre in all_genres:
        lyricists = [l for l in lyricist_songs if genre in lyricist_songs[l]]
        for i in range(len(lyricists)):
            for j in range(i + 1, len(lyricists)):
                s1 = random.choice(lyricist_songs[lyricists[i]][genre])
                s2 = random.choice(lyricist_songs[lyricists[j]][genre])
                negatives.append((s1, s2, 0))
    return negatives

def pair_key(song1, song2):
    return tuple(sorted([song1["title"], song2["title"]]))

def find_valid_genres(pairs):
    genre_labels = defaultdict(set)
    for s1, s2, label in pairs:
        for g in s1["genre"] + s2["genre"]:
            genre_labels[g].add(label)
    return {g for g, labels in genre_labels.items() if labels == {0, 1}}

# -------------------- Data Filtering and Splitting --------------------

positives = create_hard_positives()
negatives = create_hard_negatives()
all_data = positives + negatives
random.shuffle(all_data)

valid_genres = find_valid_genres(all_data)
filtered_data = [
    (s1, s2, label)
    for s1, s2, label in all_data
    if any(g in valid_genres for g in s1["genre"] + s2["genre"])
]

# Split 80:20 with no pair overlap
used = set()
train_data, test_data = [], []
target_test_size = int(0.2 * len(filtered_data))

for s1, s2, label in filtered_data:
    pk = pair_key(s1, s2)
    if pk in used:
        continue
    if len(test_data) < target_test_size:
        test_data.append((s1, s2, label))
    else:
        train_data.append((s1, s2, label))
    used.add(pk)

# -------------------- Save JSON --------------------

with open("training_data.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("testing_data_1.json", "w") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

# -------------------- Overlap Checks --------------------

def check_pair_overlap(train, test):
    train_keys = {pair_key(s1, s2) for s1, s2, _ in train}
    for s1, s2, _ in test:
        if pair_key(s1, s2) in train_keys or pair_key(s2, s1) in train_keys:
            return False
    return True

print("-------- STATS --------")
print(f"Total pairs (all_data): {len(filtered_data)}")
print(f"Training pairs: {len(train_data)}")
print(f"Test_1 pairs: {len(test_data)}")
print("Train/Test_1 pair overlap:", not check_pair_overlap(train_data, test_data))

# -------------------- Author Distribution by Genre --------------------

def compute_unique_authors_per_genre(pairs):
    genre_to_authors = defaultdict(set)
    for s1, s2, _ in pairs:
        for g in s1["genre"]:
            genre_to_authors[g].add(s1["lyricist(s)"])
        for g in s2["genre"]:
            genre_to_authors[g].add(s2["lyricist(s)"])
    return dict(sorted({g: len(a) for g, a in genre_to_authors.items()}.items(), key=lambda x: x[0]))

all_authors_dist = compute_unique_authors_per_genre(train_data + test_data)
train_authors_dist = compute_unique_authors_per_genre(train_data)
test_authors_dist = compute_unique_authors_per_genre(test_data)

# Print detailed breakdown and consistency check
print("\nGenre-wise lyricist counts (Train + Test = All):")
print(f"{'Genre':<20} {'All':>5} {'Train':>7} {'Test':>6} {'Match'}")
for genre in all_authors_dist:
    a = all_authors_dist.get(genre, 0)
    t = train_authors_dist.get(genre, 0)
    s = test_authors_dist.get(genre, 0)
    print(f"{genre:<20} {a:>5} {t:>7} {s:>6} {a == (t + s)}")

# -------------------- Plotting --------------------

def plot_hist(data, title, filename):
    genres = list(data.keys())
    counts = list(data.values())
    plt.figure(figsize=(10, 5))
    plt.bar(genres, counts)
    plt.title(title, fontproperties=chinese_font)
    plt.xlabel("流派", fontproperties=chinese_font)
    plt.ylabel("独立作词人数量", fontproperties=chinese_font)
    plt.xticks(rotation=45, ha="right", fontproperties=chinese_font)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_hist(all_authors_dist, "所有数据中的每个流派的独立作词人数量", "all_data_hist.png")
plot_hist(train_authors_dist, "训练集中的每个流派的独立作词人数量", "train_data_hist.png")
plot_hist(test_authors_dist, "测试集中的每个流派的独立作词人数量", "test_data_hist.png")

