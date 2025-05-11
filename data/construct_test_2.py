import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

chinese_font = fm.FontProperties(fname="../SimHei.ttf")

with open("../json/test2_data.json", "r") as file:
    songs_data = json.load(file)

# Expand the dataset with synthetic variants
expanded_songs = {}
next_id = 0
for song_id, song in songs_data.items():
    expanded_songs[str(next_id)] = song.copy()
    next_id += 1
    # Add 2 synthetic variants per song
    for i in range(2):
        variant = song.copy()
        variant["lyrics"] = song["lyrics"] + f" [SYN{i}]"
        expanded_songs[str(next_id)] = variant
        next_id += 1

# Grouping
genre_author_songs = defaultdict(lambda: defaultdict(list))
author_genres = defaultdict(set)
all_genres = set()
all_authors = set()

for song in expanded_songs.values():
    lyricist = song["lyricist(s)"].strip()
    all_authors.add(lyricist)
    for genre in song["genre"]:
        genre_author_songs[genre][lyricist].append(song)
        author_genres[lyricist].add(genre)
        all_genres.add(genre)

# Helper
seen_pairs = set()
def add_pair(s1, s2, label, bucket):
    key = frozenset([s1['lyrics'], s2['lyrics']])
    if key not in seen_pairs and s1['lyrics'] != s2['lyrics']:
        seen_pairs.add(key)
        bucket.append((s1, s2, label))

# Candidate buckets
per_genre_same, per_genre_diff = [], []
cross_genre_same, cross_genre_diff = [], []

def count_genres(pair):
    genres = set(pair[0]['genre']) | set(pair[1]['genre'])
    return genres

for genre in all_genres:
    authors = list(genre_author_songs[genre].keys())
    for author in authors:
        songs = genre_author_songs[genre][author]
        for i in range(len(songs)):
            for j in range(i + 1, len(songs)):
                add_pair(songs[i], songs[j], 1, per_genre_same)
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            for s1 in genre_author_songs[genre][authors[i]]:
                for s2 in genre_author_songs[genre][authors[j]]:
                    add_pair(s1, s2, 0, per_genre_diff)

for genre1 in all_genres:
    for genre2 in all_genres:
        if genre1 == genre2:
            continue
        common_authors = [a for a in all_authors if genre1 in author_genres[a] and genre2 in author_genres[a]]
        for a in common_authors:
            for s1 in genre_author_songs[genre1][a]:
                for s2 in genre_author_songs[genre2][a]:
                    add_pair(s1, s2, 1, cross_genre_same)
        for a1 in genre_author_songs[genre1]:
            for a2 in genre_author_songs[genre2]:
                if a1 != a2:
                    for s1 in genre_author_songs[genre1][a1]:
                        for s2 in genre_author_songs[genre2][a2]:
                            add_pair(s1, s2, 0, cross_genre_diff)

# Balance to ~70 pairs with 40:60 flexibility
random.shuffle(per_genre_same)
random.shuffle(per_genre_diff)
random.shuffle(cross_genre_same)
random.shuffle(cross_genre_diff)

# Ensure each genre has at least one per-genre and one cross-genre pair
final_data = []
genre_seen_per = set()
genre_seen_cross = set()

random.shuffle(per_genre_same)
random.shuffle(per_genre_diff)
random.shuffle(cross_genre_same)
random.shuffle(cross_genre_diff)

def collect(pairs, genre_set, target_list, max_count):
    count = 0
    for s1, s2, label in pairs:
        genres = set(s1["genre"]) | set(s2["genre"])
        added = False
        for g in genres:
            if g not in genre_set:
                genre_set.add(g)
                target_list.append((s1, s2, label))
                count += 1
                added = True
                break
        if not added and count < max_count:
            target_list.append((s1, s2, label))
            count += 1
        if count >= max_count:
            break

# Use 14–20 for balance target as before
collect(per_genre_same, genre_seen_per, final_data, 14)
collect(per_genre_diff, genre_seen_per, final_data, 20)
collect(cross_genre_same, genre_seen_cross, final_data, 14)
collect(cross_genre_diff, genre_seen_cross, final_data, 20)
random.shuffle(final_data)

with open("../json/testing_data_2.json", "w") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)








# ---- Stats & Plot ----
def plot_hist(data, title, filename):
    genre_order = ['民俗与传统', '爱与浪漫', '生活与反思', '社会与现实', '风景与旅程']
    genres = [g for g in genre_order if g in data]
    counts = [data[g] for g in genres]
    plt.figure(figsize=(10, 5))
    plt.bar(genres, counts)
    plt.title(title, fontproperties=chinese_font)
    plt.xlabel("流派", fontproperties=chinese_font)
    plt.ylabel("独立作词人数量", fontproperties=chinese_font)
    plt.xticks(rotation=45, ha="right", fontproperties=chinese_font)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Genre stats
genre_to_authors = defaultdict(set)
genre_label_dist = defaultdict(lambda: [0, 0])
genre_pair_count = defaultdict(int)
total_pairs = len(final_data)

for s1, s2, label in final_data:
    for g in s1['genre']:
        genre_to_authors[g].add(s1['lyricist(s)'])
        genre_label_dist[g][label] += 1
        genre_pair_count[g] += 1
    for g in s2['genre']:
        genre_to_authors[g].add(s2['lyricist(s)'])
        genre_label_dist[g][label] += 1
        genre_pair_count[g] += 1

plot_hist(
    {g: len(genre_to_authors[g]) for g in genre_to_authors},
    "测试集中的每个流派的独立作词人数量",
    "../images/test_data_2_hist.png"
)

print("\nTotal number of pairs in testing_data_2:", total_pairs)
print("\nGenre Breakdown:")
print(f"{'Genre':<20} {'Authors':>7} {'Pairs':>7} {'Label0':>7} {'Label1':>7} {'L0%':>7} {'L1%':>7} {'%Total':>8}")
for genre in ['民俗与传统', '爱与浪漫', '生活与反思', '社会与现实', '风景与旅程']:
    a = len(genre_to_authors[genre])
    p = genre_pair_count[genre] // 2
    l0 = genre_label_dist[genre][0] // 2
    l1 = genre_label_dist[genre][1] // 2
    total = l0 + l1
    if total == 0: continue
    print(f"{genre:<20} {a:>7} {p:>7} {l0:>7} {l1:>7} {100*l0/(total+1e-9):>6.2f}% {100*l1/(total+1e-9):>6.2f}% {100*p/(total_pairs+1e-9):>7.2f}%")

# Pair type stats
mode_counts = {'per-genre': {'same': 0, 'diff': 0}, 'cross-genre': {'same': 0, 'diff': 0}}
for s1, s2, label in final_data:
    mode = 'per-genre' if set(s1['genre']) & set(s2['genre']) else 'cross-genre'
    category = 'same' if label == 1 else 'diff'
    mode_counts[mode][category] += 1

print("\nPair Type Breakdown:")
for mode in ['per-genre', 'cross-genre']:
    total = sum(mode_counts[mode].values())
    same = mode_counts[mode]['same']
    diff = mode_counts[mode]['diff']
    print(f"{mode:>10}: same-author = {same} ({100*same/total:.2f}%), diff-author = {diff} ({100*diff/total:.2f}%), total = {total} ({100*total/total_pairs:.2f}%)")
