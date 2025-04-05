import os
import re
import json
import pandas as pd
from tqdm import tqdm  # Importing tqdm for the progress bar

# Function to process each file and return the processed lyrics
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = []

    for line in lines:
        # Process lyrics to remove unwanted parts
        line = re.sub(r'\[.*?\]', '', line)  # Remove content in square brackets
        line = re.sub(r'^.*?:', '', line)  # Remove text before the colon
        line = re.sub(r'[\"()~]', ' ', line)  # Replace unwanted characters with space
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

# Function to combine data and create JSON without genre
def combine_data_and_create_json(directory_path, sheet_data):
    combined_data = {}

    # Use tqdm to show progress while iterating over the rows
    for index, row in tqdm(sheet_data.iterrows(), total=sheet_data.shape[0], desc="Processing songs"):
        song_id = row['song_id']
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
    with open('songs_data_no_genre.json', 'w', encoding='utf-8') as json_file:
        json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

    print("JSON file created successfully with song data (without genre).")

# Provide the directory path where the text files are stored
directory_path = '/Users/apple/Desktop/LLM Foundations & Ethics/Dataset/lyrics'  # Change this to the path of your folder
data_file_path = '/Users/apple/Desktop/LLM Foundations & Ethics/Dataset/data.csv'  # Path to your data CSV file

# Fetch the data from CSV file
sheet_data = get_data_from_csv(data_file_path)

# Combine sheet data with processed file lengths, lyrics, and genre, then create JSON without genre generation
combine_data_and_create_json(directory_path, sheet_data)

# Calculate and print the median length of all songs
median_length, mean_length = calculate_median_length(sheet_data, directory_path)
print(f"The median song length is: {median_length}")
print(f"The mean song length is: {mean_length}")