import json
import pandas as pd

# Function to track cross-genre lyricists
def track_cross_genre(sheet_data):
    lyricist_genre_dict = {}  # Dictionary to track genres for each lyricist
    for index, row in sheet_data.iterrows():
        lyricist = row['lyricist(s)']
        
        # Ensure genre is not NaN and handle cases where 'genre' might be list-like or a string
        genre = row['genre']
        
        # If genre is a string, treat it as a list with one genre
        if isinstance(genre, str):
            genre = [genre]
        
        # If genre is NaN or not iterable (list-like), skip this row
        if isinstance(genre, (list, pd.Series)):  # Genre should be a list-like structure
            if all(pd.isna(g) for g in genre):  # Check if all elements in the list are NaN
                continue  # Skip this row if all genres are NaN
            genre = [g for g in genre if pd.notna(g)]  # Filter out NaN values in the genre list
        else:
            continue  # Skip if genre is neither string nor list-like
        
        # If genre list is empty after filtering NaN, skip the row
        if not genre:
            continue
        
        # Add all genres from the current song to the lyricist's set of genres
        if lyricist not in lyricist_genre_dict:
            lyricist_genre_dict[lyricist] = set(genre)  # Use a set to avoid duplicates
        else:
            lyricist_genre_dict[lyricist].update(genre)  # Add new genres to the existing set
    
    # Now, mark cross-genre lyricists (those who have more than one genre)
    cross_genre_lyricists = {lyricist: len(genres) > 1 for lyricist, genres in lyricist_genre_dict.items()}
    return cross_genre_lyricists

# Function to update the cross-genre status in the songs data
def update_cross_genre_status(sheet_data, cross_genre_lyricists):
    # Ensure 'cross-genre_author' column is of type bool
    sheet_data['cross-genre_author'] = sheet_data['cross-genre_author'].astype(bool)
    
    for index, row in sheet_data.iterrows():
        lyricist = row['lyricist(s)']
        # Set cross-genre status based on the lyricist's data in the dictionary
        sheet_data.at[index, 'cross-genre_author'] = bool(cross_genre_lyricists.get(lyricist, False))

    # Save the updated sheet data back to the JSON file
    with open('songs_data_filtered_Chinese.json', 'w', encoding='utf-8') as json_file:
        json.dump(sheet_data.to_dict(orient='index'), json_file, ensure_ascii=False, indent=4)
    
    print(f"Updated cross-genre_author status in 'songs_data_filtered_Chinese.json'")

# Load the existing JSON data from the file
with open('songs_data_filtered_Chinese.json', 'r', encoding='utf-8') as json_file:
    combined_data = json.load(json_file)

# Convert the JSON data to a pandas DataFrame for easy processing
sheet_data = pd.DataFrame.from_dict(combined_data, orient='index')

# Track cross-genre lyricists and get the mapping
cross_genre_lyricists = track_cross_genre(sheet_data)

# Update the cross-genre status in the DataFrame
update_cross_genre_status(sheet_data, cross_genre_lyricists)
