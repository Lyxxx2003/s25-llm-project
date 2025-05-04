"""
Not important
"""

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importing tqdm for the progress bar
import random  # For random selection of outliers to remove

# Function to process each file and return the processed lyrics
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = []

    for line in lines:
        # Process lyrics to remove unwanted parts
        line = re.sub(r'\[.*?\]', '', line)  # Remove content in square brackets
        line = re.sub(r'^.*?：', '', line)  # Remove text before the colon
        line = re.sub(r'[\"（）～【】“”()~[]"]', ' ', line)  # Replace unwanted characters with space
        line = line.strip()  # Strip any leading or trailing spaces
        
        if line:
            processed_lines.append(line)

    # Return the processed lyrics as a string and the length
    return "\n".join(processed_lines), len(processed_lines)

# Function to read the data from the CSV file
def get_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to calculate the median length of all songs
def calculate_median_length(sheet_data, directory_path):
    lengths = []  # List to store the lengths of all songs
    for index, row in sheet_data.iterrows():
        song_id = row['song_id']
        file_name = f"{song_id}.txt"  # Assuming file name matches song_id
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.exists(file_path):
            _, length = process_file(file_path)  # Get the processed length of the lyrics
            lengths.append(length)
    
    # Calculate the median
    median_length = pd.Series(lengths).median()  # Using pandas Series to calculate median
    mean_length = pd.Series(lengths).mean()
    return median_length, mean_length

# Function to combine data and create JSON without genre, excluding outliers
def combine_data_and_create_json(directory_path, sheet_data, outlier_song_ids):
    combined_data = {}

    # Use tqdm to show progress while iterating over the rows
    for index, row in tqdm(sheet_data.iterrows(), total=sheet_data.shape[0], desc="Processing songs"):
        song_id = row['song_id']
        if song_id in outlier_song_ids:
            continue  # Skip outlier songs
        
        title = row['title']
        lyricist = row['lyricist(s)']
        genre = row['genre']  # Retaining genre from the CSV
        cross_genre_author = row['cross-genre_author']
        file_name = f"{song_id}.txt"  # Assuming file name matches song_id
        
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.exists(file_path):
            lyrics, length = process_file(file_path)  # Get the processed lyrics and length
        else:
            lyrics, length = None, None
        
        # Add the song data to the combined_data dictionary
        combined_data[song_id] = {
            "title": title,
            "lyricist": lyricist,
            "genre": genre,  # Use genre from the CSV
            "cross-genre_author": cross_genre_author,
            "length": length,
            "lyrics": lyrics  # Store the processed lyrics
        }

    # Save to JSON
    with open('songs_data_no_genre_filtered_random_outliers.json', 'w', encoding='utf-8') as json_file:
        json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

    print("JSON file created successfully without randomly removed outliers.")

# Provide the directory path where the text files are stored
directory_path = './lyrics'  # Change this to the path of your folder
data_file_path = './data.csv'  # Path to your data CSV file

# Fetch the data from CSV file
sheet_data = get_data_from_csv(data_file_path)

# Calculate the song lengths
song_lengths = []
for index, row in sheet_data.iterrows():
    song_id = row['song_id']
    file_name = f"{song_id}.txt"
    file_path = os.path.join(directory_path, file_name)
    
    if os.path.exists(file_path):
        _, length = process_file(file_path)  # Get the processed length of the lyrics
        song_lengths.append(length)

# Calculate and print the median length of all songs
median_length, mean_length = calculate_median_length(sheet_data, directory_path)
print(f"The median song length is: {median_length}")
print(f"The mean song length is: {mean_length}")

# Calculate max, min
max_length = max(song_lengths)
min_length = min(song_lengths)

# Print the max and min song lengths
print(f"The maximum song length is: {max_length}")
print(f"The minimum song length is: {min_length}")

# Calculate the quartiles and IQR
Q1 = pd.Series(song_lengths).quantile(0.25)
Q3 = pd.Series(song_lengths).quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

# Identify the song IDs of outliers
outlier_song_ids = [sheet_data.iloc[index]['song_id'] for index, length in enumerate(song_lengths) 
                    if length < lower_bound or length > upper_bound]
outlier_song_ids = [int(song_id) for song_id in outlier_song_ids]

# Print the song IDs of outliers
print(f"Song IDs of outliers: {outlier_song_ids}")

# Print the number of outliers
print(f"Number of outliers: {len(outlier_song_ids)}")

# Randomly remove 3 outliers from the list
outlier_song_ids_to_remove = random.sample(outlier_song_ids, 10)
print(f"Randomly removed outliers: {outlier_song_ids_to_remove}")

# Remove the 3 outliers from the list of song_lengths
remaining_song_lengths = [length for index, length in enumerate(song_lengths) 
                          if sheet_data.iloc[index]['song_id'] not in outlier_song_ids_to_remove]

# Recalculate the IQR with the remaining song lengths
Q1_new = pd.Series(remaining_song_lengths).quantile(0.25)
Q3_new = pd.Series(remaining_song_lengths).quantile(0.75)
IQR_new = Q3_new - Q1_new

# Define new outlier bounds
lower_bound_new = Q1_new - 1.5 * IQR_new
upper_bound_new = Q3_new + 1.5 * IQR_new
print(f"New Lower Bound: {lower_bound_new}")
print(f"New Upper Bound: {upper_bound_new}")

# Regenerate the outliers based on the new bounds
remaining_outlier_song_ids = [sheet_data.iloc[index]['song_id'] for index, length in enumerate(remaining_song_lengths)
                              if length < lower_bound_new or length > upper_bound_new]

# Print the new outliers
print(f"New outliers after recalculating: {remaining_outlier_song_ids}")

# Print the number of remaining outliers
print(f"Number of remaining outliers: {len(remaining_outlier_song_ids)}")

# Combine sheet data with processed file lengths, lyrics, and genre, then create JSON without outliers
combine_data_and_create_json(directory_path, sheet_data, outlier_song_ids_to_remove)

# Find song IDs with length less than 33
short_songs_ids = [sheet_data.iloc[index]['song_id'] for index, length in enumerate(song_lengths) if length < 33]
short_songs_ids = [int(song_id) for song_id in short_songs_ids]

# Print the song IDs
print(f"Song IDs with length less than 33: {short_songs_ids}")

# Create a box plot after recalculating outliers
plt.figure(figsize=(8, 6))
plt.boxplot(remaining_song_lengths, vert=False)
plt.title('Song Length Distribution (After Removing 3 Random Outliers and Recalculating)')
plt.xlabel('Song Length (in number of lines)')
plt.show()
