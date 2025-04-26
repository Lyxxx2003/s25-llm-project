import json
import random
from collections import defaultdict

# Load the song data
with open('songs_data_filtered_Chinese.json', 'r') as file:
    songs_data = json.load(file)

# Organize songs by lyricist and genre
lyricist_songs = defaultdict(lambda: defaultdict(list))
for song_id, song_info in songs_data.items():
    lyricist = song_info['lyricist(s)']
    genres = song_info['genre']
    for genre in genres:
        lyricist_songs[lyricist][genre].append(song_info)

# Helper function to create hard positives
def create_hard_positives():
    hard_positives = []
    for lyricist, genres in lyricist_songs.items():
        if len(genres) > 1:  # Cross-genre author
            genre_pairs = [(g1, g2) for g1 in genres for g2 in genres if g1 != g2]
            for g1, g2 in genre_pairs:
                if genres[g1] and genres[g2]:
                    song1 = random.choice(genres[g1])
                    song2 = random.choice(genres[g2])
                    hard_positives.append((song1, song2, 1))  # 1 indicates same author
    return hard_positives

# Helper function to create hard negatives
def create_hard_negatives():
    hard_negatives = []
    for genre in set(g for genres in lyricist_songs.values() for g in genres):
        lyricists_in_genre = [lyricist for lyricist, genres in lyricist_songs.items() if genre in genres]
        if len(lyricists_in_genre) > 1:
            lyricist_pairs = [(l1, l2) for l1 in lyricists_in_genre for l2 in lyricists_in_genre if l1 != l2]
            for l1, l2 in lyricist_pairs:
                song1 = random.choice(lyricist_songs[l1][genre])
                song2 = random.choice(lyricist_songs[l2][genre])
                hard_negatives.append((song1, song2, 0))  # 0 indicates different authors
    return hard_negatives

# Create training and testing datasets
hard_positives = create_hard_positives()
hard_negatives = create_hard_negatives()

# Combine and shuffle
all_data = hard_positives + hard_negatives
random.shuffle(all_data)

# Select lyricists for test_2
all_lyricists = list(lyricist_songs.keys())
random.shuffle(all_lyricists)
test_2_lyricists = set(all_lyricists[:len(all_lyricists) // 5])  # Reserve 20% of lyricists for test_2

# Create training data excluding test_2 lyricists
training_data = []
for pair in all_data:
    if pair[0]['lyricist(s)'] not in test_2_lyricists:
        training_data.append(pair)

# Create test_1 and test_2 datasets
per_genre_test_1 = []
cross_genre_test_1 = []
per_genre_test_2 = []
cross_genre_test_2 = []

for pair in all_data:
    lyricist_1 = pair[0]['lyricist(s)']
    lyricist_2 = pair[1]['lyricist(s)']
    genres_1 = set(pair[0]['genre'])
    genres_2 = set(pair[1]['genre'])
    if lyricist_1 in test_2_lyricists or lyricist_2 in test_2_lyricists:
        if genres_1 & genres_2:  # Per-genre
            per_genre_test_2.append(pair)
        else:  # Cross-genre
            cross_genre_test_2.append(pair)
    else:
        if genres_1 & genres_2:  # Per-genre
            per_genre_test_1.append(pair)
        else:  # Cross-genre
            cross_genre_test_1.append(pair)

# Save the datasets
with open('training_data.json', 'w') as file:
    json.dump(training_data, file, ensure_ascii=False, indent=4)

with open('testing_data_1.json', 'w') as file:
    json.dump(per_genre_test_1 + cross_genre_test_1, file, ensure_ascii=False, indent=4)

with open('testing_data_2.json', 'w') as file:
    json.dump(per_genre_test_2 + cross_genre_test_2, file, ensure_ascii=False, indent=4) 