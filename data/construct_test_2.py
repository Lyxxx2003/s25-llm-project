import json
import random
from collections import defaultdict

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

# Category 4: Same Author, Different Genre (Increased sampling for balance)
for author in all_authors:
    author_genres_list = list(author_genres[author])
    if len(author_genres_list) >= 2:
        # Increase genre pairs per author for test_2
        max_genre_pairs = min(6, len(author_genres_list) * (len(author_genres_list) - 1) // 2)
        genre_pair_count = 0
        
        for i in range(len(author_genres_list)):
            if genre_pair_count >= max_genre_pairs:
                break
            for j in range(i + 1, len(author_genres_list)):
                if genre_pair_count >= max_genre_pairs:
                    break
                genre1, genre2 = author_genres_list[i], author_genres_list[j]
                
                # Increase songs per genre pair
                songs1 = random.sample(genre_author_songs[genre1][author], 
                                     min(4, len(genre_author_songs[genre1][author])))
                songs2 = random.sample(genre_author_songs[genre2][author], 
                                     min(4, len(genre_author_songs[genre2][author])))
                
                for s1 in songs1:
                    for s2 in songs2:
                        add_pair(s1, s2, 4, cat4_same_author_diff_genre)
                
                genre_pair_count += 1

# Category 3: Different Author, Different Genre (Increased sampling for balance)
authors_list = list(all_authors)
random.shuffle(authors_list)  # Randomize author pairs

# Increase the number of author pairs to get more Category 3 pairs
max_author_pairs = min(100, len(authors_list) * (len(authors_list) - 1) // 2)  # Increased for test_2
author_pair_count = 0

for i in range(len(authors_list)):
    if author_pair_count >= max_author_pairs:
        break
    for j in range(i + 1, len(authors_list)):
        if author_pair_count >= max_author_pairs:
            break
        author1, author2 = authors_list[i], authors_list[j]
        
        # Increase combinations for more pairs
        genre1_list = list(author_genres[author1])
        genre2_list = list(author_genres[author2])
        
        for genre1 in genre1_list:
            for genre2 in genre2_list:
                if genre1 != genre2:  # Different genres
                    # Increase songs per genre pair
                    songs1 = random.sample(genre_author_songs[genre1][author1], 
                                         min(4, len(genre_author_songs[genre1][author1])))
                    songs2 = random.sample(genre_author_songs[genre2][author2], 
                                         min(4, len(genre_author_songs[genre2][author2])))
                    
                    for s1 in songs1:
                        for s2 in songs2:
                            add_pair(s1, s2, 3, cat3_diff_author_diff_genre)
        
        author_pair_count += 1

# Balance all 4 categories to be roughly equal
random.shuffle(cat1_same_author_same_genre)
random.shuffle(cat2_diff_author_same_genre)
random.shuffle(cat3_diff_author_diff_genre)
random.shuffle(cat4_same_author_diff_genre)

# Balance all 4 categories to be roughly equal
# Find the minimum size among all categories to balance to
min_size = min(len(cat1_same_author_same_genre), len(cat2_diff_author_same_genre), 
               len(cat3_diff_author_diff_genre), len(cat4_same_author_diff_genre))

# If min_size is too small, use a reasonable target size
target_size = max(min_size, 100)  # Ensure at least 100 pairs per category for test_2

balanced_cat1 = cat1_same_author_same_genre[:min(target_size, len(cat1_same_author_same_genre))]
balanced_cat2 = cat2_diff_author_same_genre[:min(target_size, len(cat2_diff_author_same_genre))]
balanced_cat3 = cat3_diff_author_diff_genre[:min(target_size, len(cat3_diff_author_diff_genre))]
balanced_cat4 = cat4_same_author_diff_genre[:min(target_size, len(cat4_same_author_diff_genre))]

# Combine balanced categories
final_data = []
final_data.extend(balanced_cat1)
final_data.extend(balanced_cat2)
final_data.extend(balanced_cat3)
final_data.extend(balanced_cat4)

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

print(f"Total pairs in test_2 dataset: {len(final_data_enhanced)}")
