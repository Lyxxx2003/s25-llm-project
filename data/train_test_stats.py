import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set up Chinese font for matplotlib
chinese_font = fm.FontProperties(fname="../simhei.ttf")

# ------------ Load Datasets ------------

with open("../json/training_data.json", "r") as f:
    train_data = json.load(f)

with open("../json/testing_data_1.json", "r") as f:
    test_1_data = json.load(f)

with open("../json/testing_data_2.json", "r") as f:
    test_2_data = json.load(f)

# ------------ Utility Functions ------------

def pair_key(s1, s2):
    return tuple(sorted([s1["lyrics"], s2["lyrics"]]))

def check_pair_overlap(set1, set2):
    # Handle both old format (s1, s2, category) and new format (dict)
    def get_pair_key(item):
        if isinstance(item, dict):
            return pair_key(item['song1'], item['song2'])
        else:
            return pair_key(item[0], item[1])
    
    set1_keys = {get_pair_key(item) for item in set1}
    return any(get_pair_key(item) in set1_keys for item in set2)

def check_author_overlap(set1, set2):
    def get_authors(item):
        if isinstance(item, dict):
            return {item['song1']["lyricist(s)"], item['song2']["lyricist(s)"]}
        else:
            return {item[0]["lyricist(s)"], item[1]["lyricist(s)"]}
    
    authors1 = set()
    for item in set1:
        authors1.update(get_authors(item))
    
    authors2 = set()
    for item in set2:
        authors2.update(get_authors(item))
    
    return not authors1.isdisjoint(authors2)

def compute_category_distribution(pairs):
    """Compute distribution of 4 categories"""
    def get_category(item):
        if isinstance(item, dict):
            return item['category']
        else:
            return item[2]
    
    counter = Counter(int(get_category(item)) for item in pairs)
    total = sum(counter.values())
    distribution = {}
    for cat in [1, 2, 3, 4]:
        count = counter[cat]
        pct = 100 * count / total if total > 0 else 0
        distribution[cat] = (count, pct)
    return distribution, total

# ------------ Stats ------------

print("-------- OVERLAP CHECKS --------")
print("Train/Test_1 pair overlap:", check_pair_overlap(train_data, test_1_data))
print("Train/Test_2 pair overlap:", check_pair_overlap(train_data, test_2_data))
print("Train/Test_2 author overlap:", check_author_overlap(train_data, test_2_data))

print("\n-------- 4-CATEGORY DISTRIBUTION --------")
category_names = {
    1: "Same Author, Same Genre",
    2: "Diff Author, Same Genre", 
    3: "Diff Author, Diff Genre",
    4: "Same Author, Diff Genre"
}

for name, data in [("Train", train_data), ("Test_1", test_1_data), ("Test_2", test_2_data)]:
    distribution, total = compute_category_distribution(data)
    print(f"\n{name} Dataset - Total pairs: {total}")
    for cat_num in sorted(distribution.keys()):
        count, pct = distribution[cat_num]
        print(f"  Category {cat_num} ({category_names[cat_num]}): {count:>4} pairs ({pct:>5.2f}%)")

# ------------ Genre + Mode Coverage Check ------------

def validate_coverage(name, data):
    """Validate that all genres have representation in different categories"""
    genre_category_map = defaultdict(set)
    mode_counts = defaultdict(int)
    
    for item in data:
        if isinstance(item, dict):
            s1, s2, category = item['song1'], item['song2'], item['category']
            mode_counts[item['mode']] += 1
        else:
            s1, s2, category = item[0], item[1], item[2]
            # Determine mode from category for old format
            mode = "per-genre" if category in [1, 2] else "cross-genre"
            mode_counts[mode] += 1
            
        genres = set(s1["genre"]) | set(s2["genre"])
        for g in genres:
            genre_category_map[g].add(category)

    print(f"\n-------- GENRE-CATEGORY COVERAGE: {name} --------")
    for genre in sorted(genre_category_map.keys()):
        categories = sorted(genre_category_map[genre])
        print(f"{genre:<15}: Categories {categories}")
    
    print(f"\n-------- MODE DISTRIBUTION: {name} --------")
    total = sum(mode_counts.values())
    for mode, count in mode_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"{mode:<12}: {count:>4} pairs ({pct:>5.2f}%)")

# Run validation for all three sets
validate_coverage("Train", train_data)
validate_coverage("Test_1", test_1_data)
validate_coverage("Test_2", test_2_data)

# ------------ Visualization Functions ------------

def plot_genre_breakdown(data, dataset_name, filename):
    """Create genre breakdown bar chart"""
    genre_to_authors = defaultdict(set)
    genre_pair_count = defaultdict(int)
    
    for item in data:
        if isinstance(item, dict):
            s1, s2 = item['song1'], item['song2']
        else:
            s1, s2 = item[0], item[1]
            
        for g in s1['genre']:
            genre_to_authors[g].add(s1['lyricist(s)'])
            genre_pair_count[g] += 1
        for g in s2['genre']:
            genre_to_authors[g].add(s2['lyricist(s)'])
            genre_pair_count[g] += 1
    
    # Get data for plotting
    genre_order = ['民俗与传统', '爱与浪漫', '生活与反思', '社会与现实', '风景与旅程']
    genres = [g for g in genre_order if g in genre_pair_count]
    pair_counts = [genre_pair_count[g] // 2 for g in genres]  # Each pair contributes twice
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(genres, pair_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    plt.title(f'{dataset_name} - Genre Distribution (Number of Pairs)', fontproperties=chinese_font, fontsize=14)
    plt.xlabel('Genre', fontproperties=chinese_font, fontsize=12)
    plt.ylabel('Number of Pairs', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontproperties=chinese_font)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_category_distribution(data, dataset_name, filename):
    """Create 4-category distribution pie chart"""
    category_counts = defaultdict(int)
    for item in data:
        if isinstance(item, dict):
            category = item['category']
        else:
            category = item[2]
        category_counts[category] += 1
    
    category_names = {
        1: "Same Author,\nSame Genre",
        2: "Diff Author,\nSame Genre", 
        3: "Diff Author,\nDiff Genre",
        4: "Same Author,\nDiff Genre"
    }
    
    categories = []
    counts = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for cat_num in sorted(category_counts.keys()):
        if category_counts[cat_num] > 0:
            categories.append(category_names[cat_num])
            counts.append(category_counts[cat_num])
    
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(counts, labels=categories, autopct='%1.1f%%', 
                                       colors=colors[:len(counts)], startangle=90)
    
    plt.title(f'{dataset_name} - 4-Category Distribution', fontsize=14, fontweight='bold')
    
    # Enhance text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ------------ Generate Visualizations ------------

print("\n-------- GENERATING VISUALIZATIONS --------")

# Generate genre breakdown charts
plot_genre_breakdown(train_data, "Training Dataset", "../images/train_genre_dist.png")
plot_genre_breakdown(test_1_data, "Test 1 Dataset", "../images/test1_genre_dist.png") 
plot_genre_breakdown(test_2_data, "Test 2 Dataset", "../images/test2_genre_dist.png")

# Generate category distribution charts
plot_category_distribution(train_data, "Training Dataset", "../images/train_category_dist.png")
plot_category_distribution(test_1_data, "Test 1 Dataset", "../images/test1_category_dist.png")
plot_category_distribution(test_2_data, "Test 2 Dataset", "../images/test2_category_dist.png")

print("Visualizations saved to ../images/")
print("- Genre breakdown: train_genre_dist.png, test1_genre_dist.png, test2_genre_dist.png")
print("- Category distribution: train_category_dist.png, test1_category_dist.png, test2_category_dist.png")
