import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

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
    genre_to_categories = defaultdict(set)
    genre_pair_count = Counter()

    for item in data:
        if isinstance(item, dict):
            s1, s2, cat = item['song1'], item['song2'], item['category']
        else:
            s1, s2, cat = item

        genres = set(s1["genre"]) | set(s2["genre"])
        for g in genres:
            genre_to_categories[g].add(cat)
            genre_pair_count[g] += 1

    total_pairs = len(data)

    print(f"\n-------- GENRE COVERAGE SUMMARY: {name} --------")
    for genre in sorted(genre_to_categories):
        cats = sorted(genre_to_categories[genre])
        count = genre_pair_count[genre]
        pct = 100 * count / total_pairs
        print(f"{genre:<15} {cats} {count} ({pct:.2f}%)")

# Run validation for all three sets
validate_coverage("Train", train_data)
validate_coverage("Test_1", test_1_data)
validate_coverage("Test_2", test_2_data)

# ------------ Visualization Functions ------------

def plot_genre_breakdown(data, dataset_name, filename):
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

    genre_order = ['民俗与传统', '爱与浪漫', '生活与反思', '社会与现实', '风景与旅程']
    map_to_english = {
        '民俗与传统': 'Folklore & Traditional',
        '爱与浪漫': 'Love & Romance',
        '生活与反思': 'Life & Reflection',
        '社会与现实': 'Society & Reality',
        '风景与旅程': 'Scenery & Journey'
    }

    genres = [g for g in genre_order if g in genre_pair_count]
    pair_counts = [genre_pair_count[g] // 2 for g in genres]
    genres_english = [map_to_english[g] for g in genres]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(genres_english, pair_counts,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    plt.title(f'{dataset_name} - Genre Distribution (Number of Pairs)', fontsize=14)
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Number of Pairs', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def unique_lyricists_per_category(data):
    cat_to_lyricists = defaultdict(set)
    for item in data:
        if isinstance(item, dict):
            cat = item['category']
            s1, s2 = item['song1'], item['song2']
        else:
            s1, s2, cat = item
        cat_to_lyricists[cat].add(s1['lyricist(s)'])
        cat_to_lyricists[cat].add(s2['lyricist(s)'])
    return {cat: len(v) for cat, v in cat_to_lyricists.items()}

def plot_category_distribution(data, dataset_name, filename):
    category_counts = Counter(
        item['category'] if isinstance(item, dict) else item[2]
        for item in data
    )

    unique_authors = unique_lyricists_per_category(data)

    category_names = {
        1: "Same Author\nSame Genre",
        2: "Diff Author\nSame Genre",
        3: "Diff Author\nDiff Genre",
        4: "Same Author\nDiff Genre"
    }

    cats = sorted(category_counts)
    counts = [category_counts[c] for c in cats]
    labels = [category_names[c] for c in cats]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    total = sum(counts)

    plt.figure(figsize=(10, 8))
    wedges, _ = plt.pie(
        counts,
        labels=labels,
        colors=colors[:len(counts)],
        startangle=90
    )

    for wedge, cat, count in zip(wedges, cats, counts):
        angle = (wedge.theta1 + wedge.theta2) / 2
        x = 0.6 * np.cos(np.deg2rad(angle))
        y = 0.6 * np.sin(np.deg2rad(angle))

        pct = 100 * count / total
        ua = unique_authors.get(cat, 0)

        plt.text(
            x, y,
            f"{pct:.1f}%\nunique authors={ua}",
            ha='center',
            va='center',
            fontsize=10,
            color='white',
            fontweight='bold'
        )

    plt.title(f'{dataset_name} - 4-Category Distribution',
              fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ------------ Generate Visualizations ------------

print("\n-------- GENERATING VISUALIZATIONS --------")

plot_genre_breakdown(train_data, "Training Dataset", "../images/train_genre_dist.png")
plot_genre_breakdown(test_1_data, "Test 1 Dataset", "../images/test1_genre_dist.png")
plot_genre_breakdown(test_2_data, "Test 2 Dataset", "../images/test2_genre_dist.png")

plot_category_distribution(train_data, "Training Dataset", "../images/train_category_dist.png")
plot_category_distribution(test_1_data, "Test 1 Dataset", "../images/test1_category_dist.png")
plot_category_distribution(test_2_data, "Test 2 Dataset", "../images/test2_category_dist.png")

print("Visualizations saved to ../images/")
print("- Genre breakdown: train_genre_dist.png, test1_genre_dist.png, test2_genre_dist.png")
print("- Category distribution: train_category_dist.png, test1_category_dist.png, test2_category_dist.png")