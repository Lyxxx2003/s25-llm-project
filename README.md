# Chinese Lyric Authorship Verification Project

A comprehensive machine learning project focused on verifying authorship of Chinese song lyrics using Large Language Models (LLMs). This project implements both zero-shot and fine-tuning approaches to determine whether two lyrics were written by the same author, analyzing writing style across different musical genres.

## Placeholder for paper link

## Project Overview

This project tackles the challenging task of **Chinese lyric authorship verification** by:
- Analyzing writing styles across different musical genres (爱与浪漫, 生活与反思, 社会与现实, 风景与旅程, 民俗与传统)
- Implementing zero-shot evaluation using pre-trained LLMs
- Fine-tuning sentence transformers for improved authorship detection
- Evaluating performance across both per-genre and cross-genre scenarios

## Dataset

The dataset consists of Chinese song lyrics categorized into 5 main genres:
- **爱与浪漫** (Love & Romance) - 581 songs
- **生活与反思** (Life & Reflection) - 333 songs
- **风景与旅程** (Scenery & Journey) - 97 songs
- **社会与现实** (Society & Reality) - 63 songs
- **民俗与传统** (Folk & Tradition) - 27 songs

### Data Structure
- Training data: Pairs of lyrics with same/different author labels
- Test Set 1: Balanced evaluation across genres
- Test Set 2: Extended dataset with synthetic variants

## Project Structure

```
├── data/                           # Data processing and preparation
│   ├── split_train_and_test_1.py   # Train/test split with genre balancing
│   ├── construct_test_2.py         # Extended test set construction
│   ├── train_test_stats.py         # Dataset statistics
│   └── postprocess/                # Data preprocessing scripts
│       ├── post_processing_Chinese.py      # Chinese text processing
│       ├── genre_count_stats.py           # Genre distribution analysis
│       └── get_genre_of_particular_songs.py
├── zero_shot/                      # Zero-shot evaluation
│   ├── zero_shot_1.py             # Primary zero-shot experiment
│   ├── zero_shot_2.py             # Extended zero-shot experiment
│   ├── zero_shot_evaluation_1.py   # Evaluation metrics for test 1
│   └── zero_shot_evaluation_2.py   # Evaluation metrics for test 2
├── finetune/                       # Fine-tuning experiments
│   └── finetune.ipynb             # Sentence transformer fine-tuning
├── json/                          # Processed datasets
│   ├── songs_data_filtered_Chinese.json
│   ├── training_data.json
│   ├── testing_data_1.json
│   ├── testing_data_2.json
│   └── test2_data.json
├── csv/                           # Results and statistics
│   ├── genre_song_counts.csv
│   ├── zero_shot_results_df_1.csv
│   ├── zero_shot_evaluation_1_metrics_genre.csv
│   └── zero_shot_evaluation_1_metrics_mode.csv
├── images/                        # Visualizations
│   ├── genre_distribution.png
│   ├── song_length_distribution.png
│   └── *_data_hist.png
└── data/raw_lyrics/              # Raw lyric text files
    ├── train_test_1/             # Training and test set 1 lyrics
    └── test_2/                   # Test set 2 lyrics
```

## Getting Started

### Prerequisites

It's recommended to use virtual environment, below is the guide for MacOs:
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```

And then run the following command
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `together` - For LLM API access
- `scikit-learn` - Machine learning metrics
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `tqdm` - Progress bars

### Running the Experiments

#### 1. Zero-Shot Evaluation

```bash
cd zero_shot/
python zero_shot_1.py          # Run zero-shot experiment on test set 1
python zero_shot_evaluation_1.py  # Evaluate results and generate metrics
```

#### 2. Fine-tuning

Open and run the Jupyter notebook:
```bash
cd finetune/
jupyter notebook finetune.ipynb
```

#### 3. Data Processing

```bash
cd data/
python split_train_and_test_1.py    # Create balanced train/test split
python construct_test_2.py          # Generate extended test set
python train_test_stats.py          # Generate dataset statistics
```

## Results

### Zero-Shot Performance (Test Set 1)

| Genre | Mode | Accuracy | F1 Score | Precision | Recall |
|-------|------|----------|----------|-----------|---------|
| 爱与浪漫 (Love & Romance) | Per-genre | 57.1% | 56.3% | 64.7% | 61.9% |
| 生活与反思 (Life & Reflection) | Per-genre | 64.3% | 63.9% | 66.9% | 65.6% |
| 风景与旅程 (Scenery & Journey) | Per-genre | 77.8% | 67.9% | 73.3% | 66.2% |
| 民俗与传统 (Folk & Tradition) | Per-genre | 75.0% | 73.3% | 83.3% | 75.0% |
| 社会与现实 (Society & Reality) | Per-genre | 53.8% | 45.8% | 46.7% | 47.5% |

### Key Findings

1. **Genre-dependent Performance**: Folk & traditional lyrics show highest accuracy (75%), while social & reality lyrics are most challenging (53.8%)
2. **Cross-genre vs Per-genre**: Per-genre scenarios generally outperform cross-genre scenarios
3. **Style vs Content**: The model effectively distinguishes writing style from thematic content

## Technical Approach

### Zero-Shot Method
- **Model**: DeepSeek-R1-Distill-Llama-70B
- **Prompt Engineering**: Chinese language prompts focusing on linguistic features
- **Analysis Features**: Verb usage, punctuation, rare words, suffixes, humor, irony, typos

### Fine-tuning Method
- **Base Model**: Sentence Transformers
- **Loss Function**: Custom Contrastive Loss
- **Training Strategy**: Balanced sampling across genres and author pairs

### Data Preprocessing
- Chinese text normalization
- Genre classification using LLM
- Synthetic data augmentation for test set 2
- Balanced pair construction (same/different author)

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **Per-genre Analysis**: Performance breakdown by musical genre
- **Cross-genre Analysis**: Performance across different genres

## Future Work

1. **Multi-modal Analysis**: Incorporate musical features
2. **Temporal Analysis**: Consider evolution of writing style over time
3. **Author Clustering**: Group authors by similar writing styles
4. **Extended Genre Categories**: Include more nuanced genre classifications
5. **Dialectal Variations**: Account for regional Chinese language differences

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{chinese-lyric-authorship-2025,
  title={Chinese Lyric Authorship Verification using Large Language Models},
  author={Yuxin Li, Meng Fan Wang, Lorraine Xu},
  year={2025},
  url={https://github.com/Lyxxx2003/s25-llm-project}
}
```

## License

This project is licensed under the terms included in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions or collaboration opportunities, please open an issue in this repository.
