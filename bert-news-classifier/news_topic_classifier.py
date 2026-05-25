# News Topic Classifier Using BERT

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import torch
import time

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Problem Statement & Objective
print("=" * 60)
print("News Topic Classifier Using BERT")
print("=" * 60)
print("Objective: Fine-tune BERT to classify news headlines into 4 topic categories")
print("Categories: World, Sports, Business, Science/Technology")
print("=" * 60)

# Function to create synthetic dataset if Hugging Face download fails
def create_synthetic_ag_news():
    print("\nCreating synthetic dataset for demonstration...")
    
    sample_texts = {
        0: [
            "International leaders meet for peace talks in Geneva",
            "United Nations discusses climate change initiatives",
            "European Union announces new trade agreement",
            "Prime minister visits foreign country on diplomatic mission",
            "Tensions rise as regional conflict continues"
        ],
        1: [
            "Championship game ends in dramatic overtime victory",
            "Olympic athlete breaks world record in 100m sprint",
            "Football team wins league championship for third year",
            "Tennis star claims Grand Slam title at Wimbledon",
            "Basketball player scores 50 points in playoff game"
        ],
        2: [
            "Stock market reaches all-time high after positive earnings",
            "Tech company announces massive quarterly profits",
            "New business partnership drives industry growth",
            "Central bank adjusts interest rates to curb inflation",
            "Startup raises $100 million in Series B funding"
        ],
        3: [
            "Scientists develop breakthrough in artificial intelligence",
            "New discovery in quantum computing revolutionizes technology",
            "Space agency successfully launches Mars rover",
            "Researchers find cure for rare genetic disorder",
            "Renewable energy technology achieves record efficiency"
        ]
    }
    
    train_data = []
    test_data = []
    
    for label, texts in sample_texts.items():
        for i, text in enumerate(texts * 100):
            train_data.append({"text": text, "label": label})
        for i, text in enumerate(texts * 20):
            test_data.append({"text": text, "label": label})
    
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    })

# Load Dataset
print("\nLoading AG News Dataset...")
try:
    print("Attempting to download from Hugging Face...")
    dataset = hf_load_dataset("ag_news", download_mode="force_redownload")
    print(f"\nDataset loaded successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
except Exception as e:
    print(f"\nWarning: Could not download dataset from Hugging Face: {e}")
    print("Using synthetic dataset instead for demonstration purposes.")
    dataset = create_synthetic_ag_news()

# Define category labels
label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}

# Display sample data
print("\nSample data:")
for i in range(3):
    print(f"\nSample {i+1}:")
    print(f"Text: {dataset['train'][i]['text']}")
    print(f"Label: {dataset['train'][i]['label']} ({label_names[dataset['train'][i]['label']]})")

# Data Preprocessing
print("\n" + "=" * 60)
print("Tokenization & Preprocessing")
print("=" * 60)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Format datasets for PyTorch
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Create smaller subsets for faster training (optional)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(min(1000, len(tokenized_datasets["train"]))))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(min(200, len(tokenized_datasets["test"]))))

print("\nDataset preprocessing complete!")

# Model Development & Training
print("\n" + "=" * 60)
print("Model Development & Training")
print("=" * 60)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert-news-classifier",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Start training
print("\nStarting model training...")
trainer.train()

# Evaluation
print("\n" + "=" * 60)
print("Model Evaluation")
print("=" * 60)

# Evaluate on test set
print("\nEvaluating model on test set...")
eval_results = trainer.evaluate()

print("\nEvaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1-Score: {eval_results['eval_f1']:.4f}")

# Get predictions
predictions = trainer.predict(small_eval_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=list(label_names.values())))

# Plot confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(label_names.values()), 
            yticklabels=list(label_names.values()))
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Save the model and tokenizer
print("\n" + "=" * 60)
print("Saving Model")
print("=" * 60)
model.save_pretrained("./bert-news-classifier-final")
tokenizer.save_pretrained("./bert-news-classifier-final")
print("Model and tokenizer saved successfully!")

# Final Summary
print("\n" + "=" * 60)
print("Final Summary / Insights")
print("=" * 60)
print(f"✅ Successfully fine-tuned BERT on either AG News or synthetic dataset")
print(f"✅ Achieved accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"✅ Achieved F1-score: {eval_results['eval_f1']:.4f}")
print(f"✅ Model saved for deployment")
print("\nThe model can now classify news headlines into 4 categories:")
print("- World")
print("- Sports")
print("- Business")
print("- Science/Technology")
