import json
import pandas as pd

# Load the songs data from the JSON file
with open('../../json/songs_data_filtered_Chinese.json', 'r', encoding='utf-8') as json_file:
    combined_data = json.load(json_file)

# Initialize an empty dictionary to store the count of each genre
genre_counts = {}

# Define the genre concepts for reference (in English)
genre_concepts = {
    "Love & Romance": "Romance, heartbreak, longing, missing someone",
    "Life & Reflection": "Growth, regret, personal lessons, contemplation",
    "Society & Reality": "Urban struggles, inequality, class, political tone",
    "Landscape & Journey": "Nature, travel, scenery, wandering, solitude",
    "Folklore & Tradition": "Legends, cultural icons, regional storytelling, historical motifs"
}

# Iterate through the combined data to count the number of songs per genre
for song_id, song_data in combined_data.items():
    genres = song_data['genre']  # List of genres for each song
    for genre in genres:
        if genre not in genre_counts:
            genre_counts[genre] = 0
        genre_counts[genre] += 1

# Convert the genre counts into a DataFrame
table_data = []

for genre, count in genre_counts.items():
    genre_concept = genre_concepts.get(genre, "Unknown")
    table_data.append([genre, genre_concept, count])

# Create a DataFrame from the table data
df = pd.DataFrame(table_data, columns=["Genre", "Genre Concepts", "# of Songs"])

# Display the table
print(df)

# Optionally, save the table as a CSV file
df.to_csv('../../csv/genre_song_counts.csv', index=False, encoding='utf-8')