# Import necessary libraries
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load the training data
with open("training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare texts and labels
texts = []
labels = []
for entry in data:
    song1, song2, label = entry
    combined_text = song1['lyrics'] + "\n" + song2['lyrics']
    texts.append(combined_text)
    labels.append(label)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenizer and model
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Encode the datasets
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Dataset class
class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = LyricsDataset(train_encodings, train_labels)
val_dataset = LyricsDataset(val_encodings, val_labels)

# Metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {'accuracy': (preds == labels).mean()}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=2e-5,
    report_to="none",
    logging_dir="./logs",
    disable_tqdm=False,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./chinese_roberta_lyrics_finetuned")
tokenizer.save_pretrained("./chinese_roberta_lyrics_finetuned")

print("✅ Fine-tuning complete! Model saved at './chinese_roberta_lyrics_finetuned'")