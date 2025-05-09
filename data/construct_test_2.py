import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

chinese_font = fm.FontProperties(fname="../SimHei.ttf")

with open("../json/test2_data.json", "r") as file:
    songs_data = json.load(file)

# Group by genre and lyricist
genre_author_songs = defaultdict(lambda: defaultdict(list))
author_all_songs = defaultdict(list)
for song in songs_data.values():
    for genre in song["genre"]:
        genre_author_songs[genre][song["lyricist(s)"]].append(song)
    author_all_songs[song["lyricist(s)"]].append(song)

# Build pair sets
per_genre_same = []
per_genre_diff = []
cross_genre_same = []
cross_genre_diff = []
used_authors = set()

for genre, author_map in genre_author_songs.items():
    authors = list(author_map.keys())
    for author in authors:
        songs = author_map[author]
        if len(songs) >= 2:
            for i in range(len(songs)):
                for j in range(i + 1, len(songs)):
                    per_genre_same.append((songs[i], songs[j], 1))
                    used_authors.update([author])
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            s1 = random.choice(author_map[authors[i]])
            s2 = random.choice(author_map[authors[j]])
            if set(s1['genre']) & set(s2['genre']):
                per_genre_diff.append((s1, s2, 0))
                used_authors.update([authors[i], authors[j]])

for author, songs in author_all_songs.items():
    genres = defaultdict(list)
    for song in songs:
        for g in song['genre']:
            genres[g].append(song)
    g_list = list(genres.items())
    for i in range(len(g_list)):
        for j in range(i + 1, len(g_list)):
            s1_list, s2_list = g_list[i][1], g_list[j][1]
            s1 = random.choice(s1_list)
            s2 = random.choice(s2_list)
            cross_genre_same.append((s1, s2, 1))
            used_authors.add(author)

authors = list(author_all_songs.keys())
for i in range(len(authors)):
    for j in range(i + 1, len(authors)):
        a1, a2 = authors[i], authors[j]
        s1 = random.choice(author_all_songs[a1])
        s2 = random.choice(author_all_songs[a2])
        if not set(s1['genre']) & set(s2['genre']):
            cross_genre_diff.append((s1, s2, 0))
            used_authors.update([a1, a2])

# Balance and combine
random.shuffle(per_genre_same)
random.shuffle(per_genre_diff)
random.shuffle(cross_genre_same)
random.shuffle(cross_genre_diff)

per_genre_n = min(len(per_genre_same), len(per_genre_diff))
cross_genre_n = min(len(cross_genre_same), len(cross_genre_diff))
per_genre_pairs = per_genre_same[:per_genre_n] + per_genre_diff[:per_genre_n]
cross_genre_pairs = cross_genre_same[:cross_genre_n] + cross_genre_diff[:cross_genre_n]
final_data = per_genre_pairs + cross_genre_pairs
random.shuffle(final_data)

# Ensure all authors appear at least once
all_authors = set(author_all_songs.keys())
covered_authors = set()
for s1, s2, _ in final_data:
    covered_authors.add(s1['lyricist(s)'])
    covered_authors.add(s2['lyricist(s)'])
missing_authors = all_authors - covered_authors
for author in missing_authors:
    songs = author_all_songs[author]
    if len(songs) >= 2:
        final_data.append((songs[0], songs[1], 1))
    elif len(songs) == 1:
        partner_author = random.choice([a for a in author_all_songs if a != author and len(author_all_songs[a]) > 0])
        partner_song = random.choice(author_all_songs[partner_author])
        final_data.append((songs[0], partner_song, 0))

# Save output
with open("../json/testing_data_2.json", "w") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

# ================= PLOT =================
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

# ================= STATS =================
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

# ================= DISPLAY =================
print("\nTotal number of pairs in testing_data_2:", total_pairs)

print("\nGenre Breakdown:")
print(f"{'Genre':<20} {'Authors':>7} {'Pairs':>7} {'Label0':>7} {'Label1':>7} {'L0%':>7} {'L1%':>7} {'%Total':>8}")
for genre in ['民俗与传统', '爱与浪漫', '生活与反思', '社会与现实', '风景与旅程']:
    a = len(genre_to_authors[genre])
    p = genre_pair_count[genre] // 2
    l0 = genre_label_dist[genre][0] // 2
    l1 = genre_label_dist[genre][1] // 2
    total = l0 + l1
    print(f"{genre:<20} {a:>7} {p:>7} {l0:>7} {l1:>7} {100*l0/(total+1e-9):>6.2f}% {100*l1/(total+1e-9):>6.2f}% {100*p/(total_pairs+1e-9):>7.2f}%")

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
