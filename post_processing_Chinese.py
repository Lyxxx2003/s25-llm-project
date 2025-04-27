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
1. 爱与浪漫：浪漫、心碎、渴望、想念某人
2. 生活与反思：成长、遗憾、个人教训、沉思
3. 社会与现实：城市斗争、不平等、阶级、政治色彩
4. 风景与旅程：大自然、旅行、风景、漫游、孤独
5. 民俗与传统：传说、文化图标、地区故事、历史主题
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

# Function to extract genre labels from the raw API response
def extract_genre_labels(response_text):
    # Look for the "流派：" keyword and extract the genre label after it
    genre_label_prefix = "流派："
    start_index = response_text.find(genre_label_prefix) + len(genre_label_prefix)
    
    if start_index != -1:
        # Extract the genre label after "流派："
        genre_text = response_text[start_index:].strip()
        
        # Look for the genre labels inside square brackets and extract them
        if genre_text.startswith('[') and genre_text.endswith(']'):
            genre_list = genre_text[1:-1].split(',')  # Remove the brackets and split by commas
            genre_list = [genre.strip() for genre in genre_list]  # Strip extra spaces
            return genre_list
        else:
            return ["Unknown"]  # If the format is not correct, return Unknown
    else:
        return ["Unknown"]  # If "流派：" is not found in the response

# Function to use DeepSeek API to generate genre based on song lyrics
def generate_genre(lyrics, genre_concepts):
    prompt = f"""
    给定以下的流派概念：
    {genre_concepts}

    请对以下歌词进行分类：
    {lyrics}

    如果模型认为多个流派几乎同样有可能是这首歌词的流派，可以生成多个流派。但大多数情况下，应该只提供一个流派。输出应该遵循以下格式：

    流派： [流派1]

    其中，流派1是流派标签（例如，爱与浪漫，生活与反思）。如果多个流派几乎同样有可能，则应将它们列在方括号内，并用逗号分隔，例如，流派： [爱与浪漫, 生活与反思]。但请注意，大多数情况下应该只有一个流派列在方括号内。
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

    # Extract genre labels using the custom function
    genre_list = extract_genre_labels(genre_text)
    
    return genre_list

# Function to track cross-genre lyricists
def track_cross_genre(sheet_data):
    lyricist_genre_dict = {}  # Dictionary to track genres for each lyricist
    for index, row in sheet_data.iterrows():
        lyricist = row['lyricist(s)']
        
        # Ensure genre is not NaN and is a list or string (which should be a genre)
        genre = row['genre']
        
        # Skip invalid or missing genre
        if pd.isna(genre) or not isinstance(genre, list):
            continue  # Skip if genre is NaN or not a list
        
        genre = genre[0]  # Assuming 'genre' is a list and taking the first genre
        
        if lyricist not in lyricist_genre_dict:
            # If it's a new lyricist, add them with the current genre
            lyricist_genre_dict[lyricist] = [genre]
        else:
            # Check if the genre is different from what the lyricist has worked on before
            if genre not in lyricist_genre_dict[lyricist]:
                lyricist_genre_dict[lyricist].append(genre)
    
    # Now, mark cross-genre lyricists
    cross_genre_lyricists = {lyricist: len(genres) > 1 for lyricist, genres in lyricist_genre_dict.items()}
    return cross_genre_lyricists

# Function to update the cross-genre status in the songs data
def update_cross_genre_status(sheet_data, cross_genre_lyricists):
    for index, row in sheet_data.iterrows():
        lyricist = row['lyricist(s)']
        # Set cross-genre status based on the lyricist's data in the dictionary
        sheet_data.at[index, 'cross-genre_author'] = cross_genre_lyricists.get(lyricist, False)

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

# Function to combine data and create JSON, excluding outliers
def combine_data_and_create_json(directory_path, sheet_data, outlier_song_ids):
    combined_data = {}

    # Use tqdm to show progress while iterating over the rows
    for index, row in tqdm(sheet_data.iterrows(), total=sheet_data.shape[0], desc="Processing songs"):
        song_id = row['song_id']
        if song_id in outlier_song_ids:
            continue  # Skip outlier songs
        
        title = row['title']
        lyricists = row['lyricist(s)']
        genre = row['genre']
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
            "cross-genre_author": False,  # Set cross-genre_author to False initially
            "length": length,
            "lyrics": lyrics  # Store the processed lyrics
        }

    # Save to JSON
    with open('songs_data_filtered_Chinese.json', 'w', encoding='utf-8') as json_file:
        json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

    print("JSON file created successfully with all songs.")

# Function to split the data into training (80%) and testing (20%)
def split_data(input_json_path):
    # Read the input JSON file
    with open(input_json_path, 'r', encoding='utf-8') as json_file:
        combined_data = json.load(json_file)

    # Get the list of song IDs (keys of the dictionary)
    song_ids = list(combined_data.keys())
    
    # Randomly shuffle the song IDs
    random.shuffle(song_ids)
    
    # Calculate the index for the 80% training data
    train_size = int(0.8 * len(song_ids))
    
    # Split the data
    train_song_ids = song_ids[:train_size]
    test_song_ids = song_ids[train_size:]
    
    # Prepare the training and testing data
    training_data = {song_id: combined_data[song_id] for song_id in train_song_ids}
    testing_data = {song_id: combined_data[song_id] for song_id in test_song_ids}

    # Save the training data to a JSON file
    with open('training_data.json', 'w', encoding='utf-8') as json_file:
        json.dump(training_data, json_file, ensure_ascii=False, indent=4)
    
    # Save the testing data to a JSON file
    with open('testing_data.json', 'w', encoding='utf-8') as json_file:
        json.dump(testing_data, json_file, ensure_ascii=False, indent=4)

    print("Data split completed: 'training_data.json' and 'testing_data.json' created.")

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

# First pass: Create the JSON file without cross-genre_author field
combine_data_and_create_json(directory_path, sheet_data, outlier_song_ids)

# Second pass: Track cross-genre lyricists and update the cross-genre_author field
cross_genre_lyricists = track_cross_genre(sheet_data)
update_cross_genre_status(sheet_data, cross_genre_lyricists)

# Split the combined data into training and testing sets
split_data('songs_data_filtered_Chinese.json')

# Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot([length for index, length in enumerate(song_lengths) if sheet_data.iloc[index]['song_id'] not in outlier_song_ids], vert=False)
plt.title('Song Length Distribution (After Removing Outliers)')
plt.xlabel('Song Length (in number of lines)')
plt.show()