import json
import random
from collections import defaultdict

with open("../json/songs_data_filtered_Chinese.json", "r") as file:
    songs_data = json.load(file)

# Group by genre and lyricist
genre_author_songs = defaultdict(lambda: defaultdict(list))
author_all_songs = defaultdict(list)
for song in songs_data.values():
    for genre in song["genre"]:
        genre_author_songs[genre][song["lyricist(s)"]].append(song)
    author_all_songs[song["lyricist(s)"]].append(song)

# Build pair sets using 4-category system
cat1_same_author_same_genre = []  # Category 1
cat2_diff_author_same_genre = []  # Category 2
cat3_diff_author_diff_genre = []  # Category 3
cat4_same_author_diff_genre = []  # Category 4

# Helper function to avoid duplicate pairs
seen_pairs = set()
def add_pair(s1, s2, category, bucket):
    key = frozenset([s1['lyrics'], s2['lyrics']])
    if key not in seen_pairs and s1['lyrics'] != s2['lyrics']:
        seen_pairs.add(key)
        bucket.append((s1, s2, category))

# Category 1: Same Author, Same Genre
for genre, author_map in genre_author_songs.items():
    for author, songs in author_map.items():
        if len(songs) >= 2:
            for i in range(len(songs)):
                for j in range(i+1, len(songs)):
                    add_pair(songs[i], songs[j], 1, cat1_same_author_same_genre)

# Category 2: Different Author, Same Genre
for genre, author_map in genre_author_songs.items():
    authors = list(author_map.keys())
    for i in range(len(authors)):
        for j in range(i+1, len(authors)):
            for s1 in author_map[authors[i]]:
                for s2 in author_map[authors[j]]:
                    add_pair(s1, s2, 2, cat2_diff_author_same_genre)

# Category 4: Same Author, Different Genre (Increased sampling for balance)
for author, songs in author_all_songs.items():
    author_genres = defaultdict(list)
    for song in songs:
        for g in song['genre']:
            author_genres[g].append(song)
    
    genres_list = list(author_genres.keys())
    if len(genres_list) >= 2:
        # Increase combinations per author to get more Category 4 pairs
        max_genre_pairs = min(6, len(genres_list) * (len(genres_list) - 1) // 2)  # Increased from 3 to 6
        genre_pair_count = 0
        
        for i in range(len(genres_list)):
            if genre_pair_count >= max_genre_pairs:
                break
            for j in range(i+1, len(genres_list)):
                if genre_pair_count >= max_genre_pairs:  
                    break
                genre1, genre2 = genres_list[i], genres_list[j]
                
                # Increase songs per genre pair
                songs1 = random.sample(author_genres[genre1], min(4, len(author_genres[genre1])))
                songs2 = random.sample(author_genres[genre2], min(4, len(author_genres[genre2])))
                
                for s1 in songs1:
                    for s2 in songs2:
                        add_pair(s1, s2, 4, cat4_same_author_diff_genre)
                
                genre_pair_count += 1

# Category 3: Different Author, Different Genre (Increased sampling for balance)
authors = list(author_all_songs.keys())
random.shuffle(authors)  # Randomize author pairs

# Increase author pairs to get more Category 3 pairs
max_author_pairs = min(300, len(authors) * (len(authors) - 1) // 2)  # Increased significantly
author_pair_count = 0

for i in range(len(authors)):
    if author_pair_count >= max_author_pairs:
        break
    for j in range(i+1, len(authors)):
        if author_pair_count >= max_author_pairs:
            break
        author1, author2 = authors[i], authors[j]
        
        # Increase songs per author pair to get more combinations
        songs1 = random.sample(author_all_songs[author1], min(5, len(author_all_songs[author1])))
        songs2 = random.sample(author_all_songs[author2], min(5, len(author_all_songs[author2])))
        
        for s1 in songs1:
            for s2 in songs2:
                # Only include if they don't share any genres (different genre requirement)  
                if not set(s1['genre']) & set(s2['genre']):
                    add_pair(s1, s2, 3, cat3_diff_author_diff_genre)
        
        author_pair_count += 1

# ---------------- Balance 4 categories ----------------

random.shuffle(cat1_same_author_same_genre)
random.shuffle(cat2_diff_author_same_genre)
random.shuffle(cat3_diff_author_diff_genre)
random.shuffle(cat4_same_author_diff_genre)

# Balance all 4 categories to be roughly equal
# Find the minimum size among all categories to balance to
min_size = min(len(cat1_same_author_same_genre), len(cat2_diff_author_same_genre), 
               len(cat3_diff_author_diff_genre), len(cat4_same_author_diff_genre))

# If min_size is too small, use a reasonable target size
target_size = max(min_size, 500)  # Ensure at least 500 pairs per category

balanced_cat1 = cat1_same_author_same_genre[:min(target_size, len(cat1_same_author_same_genre))]
balanced_cat2 = cat2_diff_author_same_genre[:min(target_size, len(cat2_diff_author_same_genre))]
balanced_cat3 = cat3_diff_author_diff_genre[:min(target_size, len(cat3_diff_author_diff_genre))]
balanced_cat4 = cat4_same_author_diff_genre[:min(target_size, len(cat4_same_author_diff_genre))]

# Combine balanced categories
all_data = balanced_cat1 + balanced_cat2 + balanced_cat3 + balanced_cat4
random.shuffle(all_data)

# ---------------- Ensure all authors appear ----------------
all_authors = set(author_all_songs.keys())
covered_authors = set()
for s1, s2, _ in all_data:
    covered_authors.update([s1['lyricist(s)'], s2['lyricist(s)']])
missing_authors = all_authors - covered_authors
for author in missing_authors:
    songs = author_all_songs[author]
    if len(songs) >= 2:
        all_data.append((songs[0], songs[1], 1))
    elif len(songs) == 1:
        partner_author = random.choice([a for a in author_all_songs if a != author and author_all_songs[a]])
        partner_song = random.choice(author_all_songs[partner_author])
        all_data.append((songs[0], partner_song, 0))

# ---------------- Deduplicate by canonical key ----------------
def pair_key(s1, s2):
    return tuple(sorted([s1["lyrics"], s2["lyrics"]]))

deduped = list({pair_key(s1, s2): (s1, s2, label) for s1, s2, label in all_data}.items())
random.shuffle(deduped)

buckets = defaultdict(list)

for _, (s1, s2, label) in deduped:
    genres = set(s1["genre"]) | set(s2["genre"])
    mode = "per-genre" if set(s1["genre"]) & set(s2["genre"]) else "cross-genre"
    for genre in genres:
        buckets[(genre, mode)].append((s1, s2, label))

train_data, test_data = [], []
used_keys = set()

for (genre, mode), pairs in buckets.items():
    if len(pairs) < 2:
        continue  # skip underpopulated buckets

    # Deduplicate inside this bucket
    bucket_seen = set()
    clean_pairs = []
    for s1, s2, label in pairs:
        key = tuple(sorted([s1["lyrics"], s2["lyrics"]]))
        if key not in used_keys and key not in bucket_seen:
            bucket_seen.add(key)
            clean_pairs.append((s1, s2, label))

    if len(clean_pairs) < 2:
        continue

    random.shuffle(clean_pairs)
    split_idx = int(len(clean_pairs) * 0.8)

    genre_train = clean_pairs[:split_idx]
    genre_test = clean_pairs[split_idx:]

    # Record all used keys
    for s1, s2, _ in genre_train + genre_test:
        used_keys.add(tuple(sorted([s1["lyrics"], s2["lyrics"]])))

    if genre_train and genre_test:
        train_data.extend(genre_train)
        test_data.extend(genre_test)

# Optional: deduplicate again just in case
def pair_key(s1, s2):
    return tuple(sorted([s1["lyrics"], s2["lyrics"]]))

train_data = list({pair_key(s1, s2): (s1, s2, l) for s1, s2, l in train_data}.values())
test_data = list({pair_key(s1, s2): (s1, s2, l) for s1, s2, l in test_data}.values())

# ---------------- Add mode information and save ----------------
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

# Add mode information to datasets
train_data_enhanced = add_mode_info(train_data)
test_data_enhanced = add_mode_info(test_data)

# Randomly sample desired amounts
random.shuffle(train_data_enhanced)
random.shuffle(test_data_enhanced)

# Sample 6000 training pairs and 1500 test pairs
target_train_size = 6000
target_test_size = 1500

sampled_train_data = train_data_enhanced[:min(target_train_size, len(train_data_enhanced))]
sampled_test_data = test_data_enhanced[:min(target_test_size, len(test_data_enhanced))]

# Save sampled data
with open("../json/training_data.json", "w") as f:
    json.dump(sampled_train_data, f, ensure_ascii=False, indent=2)

with open("../json/testing_data_1.json", "w") as f:
    json.dump(sampled_test_data, f, ensure_ascii=False, indent=2)

print(f"Final training data size: {len(sampled_train_data)} pairs")
print(f"Final testing data size: {len(sampled_test_data)} pairs")