import json
from collections import defaultdict, Counter

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
    set1_keys = {pair_key(s1, s2) for s1, s2, _ in set1}
    return any(pair_key(s1, s2) in set1_keys for s1, s2, _ in set2)

def check_author_overlap(set1, set2):
    authors1 = {s["lyricist(s)"] for pair in set1 for s in pair[:2]}
    authors2 = {s["lyricist(s)"] for pair in set2 for s in pair[:2]}
    return not authors1.isdisjoint(authors2)

def compute_label_distribution(pairs):
    counter = Counter(int(label) for _, _, label in pairs)
    total = counter[0] + counter[1]
    pct_0 = 100 * counter[0] / total if total > 0 else 0
    pct_1 = 100 * counter[1] / total if total > 0 else 0
    return counter[0], counter[1], pct_0, pct_1

# ------------ Stats ------------

print("-------- OVERLAP CHECKS --------")
print("Train/Test_1 pair overlap:", check_pair_overlap(train_data, test_1_data))
print("Train/Test_2 pair overlap:", check_pair_overlap(train_data, test_2_data))
print("Train/Test_2 author overlap:", check_author_overlap(train_data, test_2_data))

print("\n-------- LABEL DISTRIBUTION --------")
for name, data in [("Train", train_data), ("Test_1", test_1_data), ("Test_2", test_2_data)]:
    l0, l1, p0, p1 = compute_label_distribution(data)
    print(f"{name:<7}: Label 0 = {l0:>4} ({p0:>5.2f}%), Label 1 = {l1:>4} ({p1:>5.2f}%), Total = {l0 + l1}")

# ------------ Genre + Mode Coverage Check ------------

def validate_coverage(name, data):
    genre_mode_map = defaultdict(set)
    for s1, s2, _ in data:
        genres = set(s1["genre"]) | set(s2["genre"])
        mode = "per-genre" if set(s1["genre"]) & set(s2["genre"]) else "cross-genre"
        for g in genres:
            genre_mode_map[g].add(mode)

    print(f"\n-------- GENRE-MODE COVERAGE: {name} --------")
    for genre in sorted(genre_mode_map.keys()):
        modes = genre_mode_map[genre]
        has_per = "per-genre" in modes
        has_cross = "cross-genre" in modes
        status = "✅" if has_per and has_cross else "⚠️ Missing:"
        detail = []
        if not has_per: detail.append("per-genre")
        if not has_cross: detail.append("cross-genre")
        print(f"{genre:<10}: {status} {' & '.join(detail) if detail else ''}")

# Run validation for all three sets
validate_coverage("Train", train_data)
validate_coverage("Test_1", test_1_data)
validate_coverage("Test_2", test_2_data)
