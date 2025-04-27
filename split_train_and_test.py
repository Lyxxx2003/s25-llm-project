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
        if len(genres) > 1:
            genre_pairs = [(g1, g2) for g1 in genres for g2 in genres if g1 != g2]
            for g1, g2 in genre_pairs:
                if genres[g1] and genres[g2]:
                    song1 = random.choice(genres[g1])
                    song2 = random.choice(genres[g2])
                    hard_positives.append((song1, song2, 1))
    return hard_positives

# Helper function to create hard negatives
def create_hard_negatives():
    hard_negatives = []
    for genre in set(g for genres in lyricist_songs.values() for g in genres):
        lyricists_in_genre = [lyricist for lyricist, genres in lyricist_songs.items() if genre in genres]
        if len(lyricists_in_genre) > 1:
            lyricist_pairs = [(l1, l2) for i, l1 in enumerate(lyricists_in_genre) for l2 in lyricists_in_genre[i+1:]]
            for l1, l2 in lyricist_pairs:
                song1 = random.choice(lyricist_songs[l1][genre])
                song2 = random.choice(lyricist_songs[l2][genre])
                hard_negatives.append((song1, song2, 0))
    return hard_negatives

# Create pairs
hard_positives = create_hard_positives()
hard_negatives = create_hard_negatives()
all_data = hard_positives + hard_negatives
random.shuffle(all_data)

# Identify valid genres with both 0 and 1 labels
def find_valid_genres(pairs):
    genre_labels = defaultdict(set)
    for song1, song2, label in pairs:
        for genre in song1['genre'] + song2['genre']:
            genre_labels[genre].add(label)
    return {genre for genre, labels in genre_labels.items() if labels == {0, 1}}

valid_genres = find_valid_genres(all_data)

# Filter pairs to only valid genres
filtered_data = []
for pair in all_data:
    song1, song2, label = pair
    if any(genre in valid_genres for genre in song1['genre'] + song2['genre']):
        filtered_data.append(pair)

all_data = filtered_data

# Identify all unique lyricists
all_lyricists = list(set(song['lyricist(s)'] for pair in all_data for song in pair[:2]))
random.shuffle(all_lyricists)

# Reserve some lyricists completely for test_2
num_test_2_lyricists = max(1, int(0.1 * len(all_lyricists)))
reserved_lyricists = set(all_lyricists[:num_test_2_lyricists])

# Prepare allocation sets
used_pairs = set()
test_2_data = []
test_1_data = []
training_data = []

# Helper to create unique pair key
def pair_key(song1, song2):
    return tuple(sorted([song1['title'], song2['title']]))

# First assign test_2 strictly
for pair in all_data:
    song1, song2, label = pair
    lyricist1 = song1['lyricist(s)']
    lyricist2 = song2['lyricist(s)']
    pk = pair_key(song1, song2)
    if lyricist1 in reserved_lyricists and lyricist2 in reserved_lyricists:
        if pk not in used_pairs:
            test_2_data.append(pair)
            used_pairs.add(pk)

# Then assign test_1 and train
for pair in all_data:
    song1, song2, label = pair
    lyricist1 = song1['lyricist(s)']
    lyricist2 = song2['lyricist(s)']
    pk = pair_key(song1, song2)
    if pk in used_pairs:
        continue
    if lyricist1 in reserved_lyricists or lyricist2 in reserved_lyricists:
        continue
    if len(test_1_data) < int(0.1 * len(all_data)):
        test_1_data.append(pair)
    else:
        training_data.append(pair)
    used_pairs.add(pk)

# Save datasets
with open('training_data.json', 'w') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=4)

with open('testing_data_1.json', 'w') as f:
    json.dump(test_1_data, f, ensure_ascii=False, indent=4)

with open('testing_data_2.json', 'w') as f:
    json.dump(test_2_data, f, ensure_ascii=False, indent=4)

# Print stats
print(f"Total pairs after filtering: {len(all_data)}")
print(f"Training samples: {len(training_data)}")
print(f"Test_1 samples: {len(test_1_data)}")
print(f"Test_2 samples: {len(test_2_data)}")

# Check for overlaps
def check_pair_overlap(train, test):
    train_pairs = set()
    for pair in train:
        train_pairs.add(pair_key(pair[0], pair[1]))

    for pair in test:
        if pair_key(pair[0], pair[1]) in train_pairs:
            return False
    return True

def check_author_overlap(train, test):
    train_authors = set()
    for pair in train:
        train_authors.add(pair[0]['lyricist(s)'])
        train_authors.add(pair[1]['lyricist(s)'])

    for pair in test:
        if pair[0]['lyricist(s)'] in train_authors or pair[1]['lyricist(s)'] in train_authors:
            return False
    return True

print("Train/Test_1 pair overlap:", not check_pair_overlap(training_data, test_1_data))
print("Train/Test_2 pair overlap:", not check_pair_overlap(training_data, test_2_data))
print("Train/Test_2 author overlap:", not check_author_overlap(training_data, test_2_data))