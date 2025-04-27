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

# Identify all unique lyricists
all_lyricists = list(lyricist_songs.keys())
random.shuffle(all_lyricists)

# Reserve some lyricists completely for test_2
num_test_2_lyricists = max(1, int(0.1 * len(all_lyricists)))
reserved_lyricists = set(all_lyricists[:num_test_2_lyricists])

# Split data into test_2, test_1 candidates, and training candidates
test_2_data = []
test_1_candidates = []
train_candidates = []
for pair in all_data:
    lyricist_1 = pair[0]['lyricist(s)']
    lyricist_2 = pair[1]['lyricist(s)']
    if lyricist_1 in reserved_lyricists or lyricist_2 in reserved_lyricists:
        if lyricist_1 in reserved_lyricists and lyricist_2 in reserved_lyricists:
            test_2_data.append(pair)
    else:
        test_1_candidates.append(pair)

random.shuffle(test_1_candidates)

# Calculate target sizes
total_samples = len(all_data)
train_size = int(0.8 * total_samples)
test_total_size = total_samples - train_size
test_1_size = test_total_size - len(test_2_data)

if test_1_size < 0:
    raise ValueError("Test_2 is too large, cannot satisfy 80/20 split!")

# Split test_1 and training data
test_1_data = test_1_candidates[:test_1_size]
training_data = test_1_candidates[test_1_size:]

# Save datasets
with open('training_data.json', 'w') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=4)

with open('testing_data_1.json', 'w') as f:
    json.dump(test_1_data, f, ensure_ascii=False, indent=4)

with open('testing_data_2.json', 'w') as f:
    json.dump(test_2_data, f, ensure_ascii=False, indent=4)

# Print stats
print(f"Total samples: {len(all_data)}")
print(f"Training samples: {len(training_data)}")
print(f"Test_1 samples: {len(test_1_data)}")
print(f"Test_2 samples: {len(test_2_data)}")

# Check for overlaps
def check_song_overlap(train, test):
    train_titles = set()
    for pair in train:
        train_titles.add(pair[0]['title'])
        train_titles.add(pair[1]['title'])

    for pair in test:
        if pair[0]['title'] in train_titles or pair[1]['title'] in train_titles:
            return False
    return True

print("Train/Test_1 song overlap:", not check_song_overlap(training_data, test_1_data))
print("Train/Test_2 song overlap:", not check_song_overlap(training_data, test_2_data))

def check_author_overlap(train, test):
    train_authors = set()
    for pair in train:
        train_authors.add(pair[0]['lyricist(s)'])
        train_authors.add(pair[1]['lyricist(s)'])

    for pair in test:
        if pair[0]['lyricist(s)'] in train_authors or pair[1]['lyricist(s)'] in train_authors:
            return False
    return True

print("Train/Test_2 author overlap:", not check_author_overlap(training_data, test_2_data))