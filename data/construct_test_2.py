import json
import random
from collections import defaultdict
from sklearn.utils import resample
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Chinese font configuration
chinese_font = fm.FontProperties(fname="../SimHei.ttf")

# ======== DATA LOADING & PREPROCESSING ========
with open("../json/test2_data.json", "r") as file:
    songs_data = json.load(file)

# Data structures
genre_author_songs = defaultdict(lambda: defaultdict(list))
author_genres = defaultdict(set)
all_genres = set()
all_authors = set()

for song in songs_data.values():
    lyricist = song["lyricist(s)"].strip()
    all_authors.add(lyricist)
    for genre in song["genre"]:
        genre_author_songs[genre][lyricist].append(song)
        author_genres[lyricist].add(genre)
        all_genres.add(genre)

# ======== PAIR GENERATION CORE ========
seen_pairs = set()
final_data = []
MAX_ATTEMPTS = 50  # Safety limit for pair generation

def add_pair(s1, s2, label):
    """Safe pair addition with duplicate check"""
    key = frozenset([s1['lyrics'], s2['lyrics']])
    if key not in seen_pairs and s1['lyrics'] != s2['lyrics']:
        seen_pairs.add(key)
        final_data.append((s1, s2, label))
        return True
    return False

# ======== PER-GENRE PAIR GENERATION ========
MIN_PAIRS_PER_GENRE = 2  # Minimum pairs per label per genre

for genre in all_genres:
    authors = list(genre_author_songs[genre].keys())
    if not authors:
        continue

    # Track generation attempts
    attempts = {'same': 0, 'diff': 0}
    
    # Generate same-author pairs (Label 1)
    same_count = 0
    for author in authors:
        songs = genre_author_songs[genre][author]
        if len(songs) >= 2:
            for i in range(len(songs)):
                for j in range(i+1, len(songs)):
                    if add_pair(songs[i], songs[j], 1):
                        same_count += 1

    # Fallback for insufficient same-author pairs
    while same_count < MIN_PAIRS_PER_GENRE and attempts['same'] < MAX_ATTEMPTS:
        author = random.choice(authors)
        songs = genre_author_songs[genre][author]
        if len(songs) >= 1:
            other_authors = [a for a in authors if a != author]
            if other_authors:
                partner_author = random.choice(other_authors)
                partner_songs = genre_author_songs[genre][partner_author]
                if partner_songs:
                    added = add_pair(
                        random.choice(songs),
                        random.choice(partner_songs),
                        1
                    )
                    if added: same_count += 1
        attempts['same'] += 1

    # Generate diff-author pairs (Label 0)
    diff_count = 0
    if len(authors) >= 2:
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                a1_songs = genre_author_songs[genre][authors[i]]
                a2_songs = genre_author_songs[genre][authors[j]]
                for s1 in a1_songs:
                    for s2 in a2_songs:
                        if add_pair(s1, s2, 0):
                            diff_count += 1

    # Fallback for insufficient diff-author pairs
    while diff_count < MIN_PAIRS_PER_GENRE and attempts['diff'] < MAX_ATTEMPTS:
        if len(authors) >= 2:
            a1, a2 = random.sample(authors, 2)
            s1 = random.choice(genre_author_songs[genre][a1]) if genre_author_songs[genre][a1] else None
            s2 = random.choice(genre_author_songs[genre][a2]) if genre_author_songs[genre][a2] else None
            if s1 and s2 and add_pair(s1, s2, 0):
                diff_count += 1
        attempts['diff'] += 1

# ======== CROSS-GENRE PAIR GENERATION ========
cross_genre_pairs = []
processed_pairs = set()

for genre1 in all_genres:
    for genre2 in all_genres:
        if genre1 == genre2:
            continue
        
        # Cross-genre same-author pairs
        common_authors = [
            a for a in all_authors
            if genre1 in author_genres[a] and genre2 in author_genres[a]
        ]
        for author in common_authors:
            songs1 = genre_author_songs[genre1][author]
            songs2 = genre_author_songs[genre2][author]
            for s1 in songs1:
                for s2 in songs2:
                    if add_pair(s1, s2, 1):
                        cross_genre_pairs.append((s1, s2, 1))

        # Cross-genre diff-author pairs
        authors1 = list(genre_author_songs[genre1].keys())
        authors2 = list(genre_author_songs[genre2].keys())
        for a1 in authors1:
            for a2 in authors2:
                if a1 != a2:
                    s1 = random.choice(genre_author_songs[genre1][a1])
                    s2 = random.choice(genre_author_songs[genre2][a2])
                    if add_pair(s1, s2, 0):
                        cross_genre_pairs.append((s1, s2, 0))

# ======== FINAL BALANCING ========
# Balance labels globally
same_pairs = [p for p in final_data if p[2] == 1]
diff_pairs = [p for p in final_data if p[2] == 0]
min_count = min(len(same_pairs), len(diff_pairs))

if min_count > 0:
    balanced_data = (
        resample(same_pairs, n_samples=min_count, random_state=42) +
        resample(diff_pairs, n_samples=min_count, random_state=42)
    )
    random.shuffle(balanced_data)
    final_data = balanced_data
else:
    print("Warning: Insufficient data for label balancing")

# Save dataset
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
