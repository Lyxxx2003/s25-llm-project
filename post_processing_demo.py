"""
Not important
"""

from together import Together
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importing tqdm for the progress bar
import random  # For random selection of outliers to remove

# Initialize the API client
client = Together(api_key='95f8a868b1f39901f4ee26ab1e582d980a987cea980b1259edcb68b62d9a851c')  # Make sure to set the correct API key

# Define the genre concepts for the model to use
genre_concepts = """
1. Love & Romance: Romance, heartbreak, longing, missing someone
2. Life & Reflection: Growth, regret, personal lessons, contemplation
3. Society & Reality: Urban struggles, inequality, class, political tone
4. Landscape & Journey: Nature, travel, scenery, wandering, solitude
5. Folklore & Tradition: Legends, cultural icons, regional storytelling, historical motifs
"""

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

# Function to use DeepSeek API to generate genre based on song lyrics
def generate_genre(lyrics, genre_concepts):
    prompt = f"""
    Given the following genre concepts:
    {genre_concepts}

    Please classify the genre of the following lyrics:
    {lyrics}

    If the model thinks multiple genres are equally likely to be the genre of these lyrics, it can generate multiple genres. However, in most cases, only one genre should be provided. The output should follow the format:

    Genres: [Genre1]

    Where Genre1 is the genre label (e.g., Love & Romance, Life & Reflection). If multiple genres are equally likely, they should be listed inside the square brackets and separated by commas, e.g., Genres: [Love & Romance, Life & Reflection]. But again, most of the time, there should only be one genre listed inside the brackets.
    """
    
    print("Making API request...")  # Debugging print to see if it's reaching here
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        messages=[{"role": "user", "content": prompt}],
    )
    print("API request completed.")  # Debugging print to see if the request was successful
    
    # Debugging the API response
    genre_text = response.choices[0].message.content.strip()
    print(f"Raw API Response: {genre_text}")  # Output the raw response for debugging

    # Extract the genre from between '**'
    start_index = genre_text.find("**") + 2  # Find the first '**'
    end_index = genre_text.find("**", start_index)  # Find the second '**'
    
    # If both '**' are found, extract the genre between them
    if start_index != -1 and end_index != -1:
        genre = genre_text[start_index:end_index].strip()
    else:
        genre = "Unknown"  # Fallback if the genre format is incorrect

    # Ensure that we have a valid genre label
    if not genre:
        print("Error: No valid genre label found.")
        return "Unknown"  # Return a default value if no valid genre is found
    
    return genre

# Function to remove outliers based on IQR
def remove_outliers(sheet_data, song_lengths):
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
    
    # Return the list of outlier song IDs
    return outlier_song_ids

# Function to combine data and create JSON for the first 10 songs, excluding outliers
def combine_data_and_create_json(directory_path, sheet_data, outlier_song_ids):
    combined_data = {}

    # Use tqdm to show progress while iterating over the rows
    for index, row in tqdm(sheet_data.head(10).iterrows(), total=10, desc="Processing first 10 songs"):
        song_id = row['song_id']
        if song_id in outlier_song_ids:
            continue  # Skip outlier songs
        
        title = row['title']
        lyricists = row['lyricist(s)']
        genre = row['genre']
        cross_genre_author = row['cross-genre_author']
        file_name = f"{song_id}.txt"  # Assuming file name matches song_id
        
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.exists(file_path):
            lyrics, length = process_file(file_path)  # Get the processed lyrics and length
            generated_genre = generate_genre(lyrics, genre_concepts)  # Get the genre from DeepSeek
        else:
            lyrics, length, generated_genre = None, None, None
        
        # Add the song data to the combined_data dictionary
        combined_data[song_id] = {
            "title": title,
            "lyricist(s)": lyricists,
            "genre": generated_genre if generated_genre else genre,  # Use generated genre if available
            "cross-genre_author": cross_genre_author,
            "length": length,
            "lyrics": lyrics  # Store the processed lyrics
        }

    # Save to JSON
    with open('songs_data_first_10_filtered.json', 'w', encoding='utf-8') as json_file:
        json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

    print("JSON file created successfully with first 10 songs.")

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

# Remove outliers and get the updated list
outlier_song_ids = remove_outliers(sheet_data, song_lengths)

# Calculate and print the median length of all songs (after removing outliers)
median_length, mean_length = calculate_median_length(sheet_data, directory_path)
print(f"The median song length is: {median_length}")
print(f"The mean song length is: {mean_length}")

# Calculate max, min
max_length = max(song_lengths)
min_length = min(song_lengths)

# Print the max and min song lengths
print(f"The maximum song length is: {max_length}")
print(f"The minimum song length is: {min_length}")

# Combine sheet data with processed file lengths, lyrics, and generated genres, then create JSON for the first 10 songs
combine_data_and_create_json(directory_path, sheet_data, outlier_song_ids)

# Find song IDs with length less than 33
short_songs_ids = [sheet_data.iloc[index]['song_id'] for index, length in enumerate(song_lengths) if length < 33]
short_songs_ids = [int(song_id) for song_id in short_songs_ids]

# Print the song IDs
print(f"Song IDs with length less than 33: {short_songs_ids}")

# Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot([length for index, length in enumerate(song_lengths) if sheet_data.iloc[index]['song_id'] not in outlier_song_ids], vert=False)
plt.title('Song Length Distribution (After Removing Outliers)')
plt.xlabel('Song Length (in number of lines)')
plt.show()
