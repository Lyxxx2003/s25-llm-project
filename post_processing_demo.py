from together import Together
import os
import re
import json
import pandas as pd
from tqdm import tqdm  # Importing tqdm for the progress bar
import time  # To measure the time taken for API requests

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

# Function to use DeepSeek API to generate genre based on song lyrics
def generate_genre(lyrics, genre_concepts):
    prompt = f"Given the following genre concepts:\n{genre_concepts}\n\nClassify the genre of the following lyrics. ONLY return the genre label. Do not provide any explanations, reasoning, or extra text. Just the genre label. Do not write anything else:\n\n{lyrics}\n\nGenre:"
    
    print("Making API request...")  # Debugging print to see if it's reaching here
    start_time = time.time()  # Start the timer
    
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        messages=[{"role": "user", "content": prompt}],
    )
    
    end_time = time.time()  # End the timer
    print(f"API request completed in {end_time - start_time:.2f} seconds.")  # Log the time taken
    
    # Debugging the API response
    print(f"API Response: {response.choices[0].message.content.strip()}")
    
    genre = response.choices[0].message.content.strip()
    return genre

# Function to combine data and create JSON for the first 10 songs with progress bar
def combine_data_and_create_json(directory_path, sheet_data, num_songs=1):
    combined_data = {}

    # Use tqdm to show progress while iterating over the first 10 rows
    for index, row in tqdm(sheet_data.head(num_songs).iterrows(), total=num_songs, desc="Processing 10 songs"):
        song_id = row['song_id']
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

    # Save to a new JSON file with the first 10 songs' data
    with open('songs_data_first_10.json', 'w', encoding='utf-8') as json_file:
        json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

    print("JSON file created successfully with first 10 songs.")

# Provide the directory path where the text files are stored
directory_path = '/Users/apple/Desktop/LLM Foundations & Ethics/Dataset/lyrics'  # Change this to the path of your folder
data_file_path = '/Users/apple/Desktop/LLM Foundations & Ethics/Dataset/data.csv'  # Path to your data CSV file

# Fetch the data from CSV file
sheet_data = get_data_from_csv(data_file_path)

# Combine sheet data with processed file lengths, lyrics, and generated genres, then create JSON for the first 10 songs
combine_data_and_create_json(directory_path, sheet_data, num_songs=1)
