import json
import random
from collections import defaultdict
import matplotlib.font_manager as fm

chinese_font = fm.FontProperties(fname="../SimHei.ttf")

# Load test2_data.json directly (already verified to have no author overlap with train/test_1)
with open("../json/test2_data.json", "r") as file:
    songs_data = json.load(file)

# Grouping
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

# Generate pairs using 4-category system
seen_pairs = set()
def add_pair(s1, s2, category, bucket):
    key = frozenset([s1['lyrics'], s2['lyrics']])
    if key not in seen_pairs and s1['lyrics'] != s2['lyrics']:
        seen_pairs.add(key)
        bucket.append((s1, s2, category))

# 4 category buckets
cat1_same_author_same_genre = []  # Category 1
cat2_diff_author_same_genre = []  # Category 2
cat3_diff_author_diff_genre = []  # Category 3
cat4_same_author_diff_genre = []  # Category 4

# Category 1: Same Author, Same Genre
for genre in all_genres:
    authors = list(genre_author_songs[genre].keys())
    for author in authors:
        songs = genre_author_songs[genre][author]
        if len(songs) >= 2:
            for i in range(len(songs)):
                for j in range(i + 1, len(songs)):
                    add_pair(songs[i], songs[j], 1, cat1_same_author_same_genre)

# Category 2: Different Author, Same Genre  
for genre in all_genres:
    authors = list(genre_author_songs[genre].keys())
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            for s1 in genre_author_songs[genre][authors[i]]:
                for s2 in genre_author_songs[genre][authors[j]]:
                    add_pair(s1, s2, 2, cat2_diff_author_same_genre)

# Category 4: Same Author, Different Genre
for author in all_authors:
    author_genres_list = list(author_genres[author])
    if len(author_genres_list) >= 2:
        for i in range(len(author_genres_list)):
            for j in range(i + 1, len(author_genres_list)):
                genre1, genre2 = author_genres_list[i], author_genres_list[j]
                for s1 in genre_author_songs[genre1][author]:
                    for s2 in genre_author_songs[genre2][author]:
                        add_pair(s1, s2, 4, cat4_same_author_diff_genre)

# Category 3: Different Author, Different Genre
authors_list = list(all_authors)
for i in range(len(authors_list)):
    for j in range(i + 1, len(authors_list)):
        author1, author2 = authors_list[i], authors_list[j]
        for genre1 in author_genres[author1]:
            for genre2 in author_genres[author2]:
                if genre1 != genre2:  # Different genres
                    for s1 in genre_author_songs[genre1][author1]:
                        for s2 in genre_author_songs[genre2][author2]:
                            add_pair(s1, s2, 3, cat3_diff_author_diff_genre)

# Balance categories - aim for roughly equal representation
random.shuffle(cat1_same_author_same_genre)
random.shuffle(cat2_diff_author_same_genre)
random.shuffle(cat3_diff_author_diff_genre)
random.shuffle(cat4_same_author_diff_genre)

# Combine all categories without forcing equal distribution
# This will create a natural distribution similar to train/test_1
final_data = []
final_data.extend(cat1_same_author_same_genre)
final_data.extend(cat2_diff_author_same_genre)
final_data.extend(cat3_diff_author_diff_genre)
final_data.extend(cat4_same_author_diff_genre)

random.shuffle(final_data)

# Add mode information and save
def add_mode_info(data):
    """Add mode information based on category: 1,2 = per-genre; 3,4 = cross-genre"""
    enhanced_data = []
    for s1, s2, category in data:
        # Determine mode based on category
        if category in [1, 2]:
            mode = "per-genre"
        else:  # category in [3, 4]
            mode = "cross-genre"
        
        # Create enhanced entry with mode info
        enhanced_entry = {
            "song1": s1,
            "song2": s2, 
            "category": category,
            "mode": mode,
            "label": 1 if category in [1, 4] else 0  # Same author = 1, Diff author = 0
        }
        enhanced_data.append(enhanced_entry)
    return enhanced_data

final_data_enhanced = add_mode_info(final_data)

with open("../json/testing_data_2.json", "w") as f:
    json.dump(final_data_enhanced, f, ensure_ascii=False, indent=2)

print(f"Saved {len(final_data_enhanced)} pairs to testing_data_2.json")
