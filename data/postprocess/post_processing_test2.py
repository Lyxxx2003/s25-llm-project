import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importing tqdm for the progress bar

# Function to process each file and return the processed lyrics
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = []

    for line in lines:
        # Process lyrics to remove unwanted parts
        line = re.sub(r'\[.*?\]', '', line)  # Remove content in square brackets
        line = re.sub(r'^.*?：', '', line)  # Remove text before the colon
        # Replace unwanted characters with space
        line = re.sub(r'[\"（）～【】“”()~[\]]', ' ', line)  # Add more unwanted characters here
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
        # if song_id in outlier_song_ids:
        #     continue  # Skip outlier songs
        
        title = row['title']
        lyricist = row['lyricist(s)']
        genre = row['genre']  # Retaining genre from the CSV
        file_name = f"{song_id}.txt"  # Assuming file name matches song_id
        
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.exists(file_path):
            lyrics, length = process_file(file_path)  # Get the processed lyrics and length
        else:
            lyrics, length = None, None
        
        # Add the song data to the combined_data dictionary
        combined_data[song_id] = {
            "title": title,
            "lyricist(s)": lyricist,
            "genre": [genre],  # Use genre from the CSV
            "length": length,
            "lyrics": lyrics  # Store the processed lyrics
        }

    # Save to JSON
    with open('json/testing_data_2.json', 'w', encoding='utf-8') as json_file:
        json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

    print("JSON file created successfully without outliers.")

# Provide the directory path where the text files are stored
directory_path = './test2_lyrics'  # Change this to the path of your folder
data_file_path = 'csv/test2_data.csv'  # Path to your data CSV file

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

# Combine sheet data with processed file lengths, lyrics, and genre, then create JSON without outliers
combine_data_and_create_json(directory_path, sheet_data, outlier_song_ids)

# Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(song_lengths, vert=False)
plt.title('Song Length Distribution')
plt.xlabel('Song Length (in number of lines)')
plt.show()
