"""
Fine-tuning BERT (bert-base-uncased) for Academic Intent Classification in EduMind.
Demonstrates LLM Transformer fine-tuning using PyTorch and Hugging Face Transformers.
"""

import os
import json
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "classification/model/bert_finetuned"
RESULTS_DIR = "evaluation/results"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
MAX_LENGTH = 64

class CollegeIntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def get_device():
    if torch.backends.mps.is_available():
        try:
            # Quick check if MPS works for basic tensor
            t = torch.zeros(1).to("mps")
            return torch.device("mps")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_bert():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = get_device()
    print(f"Training on compute device: {device}")

    # Load data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")

    unique_intents = sorted(train_df["intent"].unique())
    label2id = {intent: i for i, intent in enumerate(unique_intents)}
    id2label = {i: intent for intent, i in label2id.items()}

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    train_labels = [label2id[i] for i in train_df["intent"]]
    val_labels = [label2id[i] for i in val_df["intent"]]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = CollegeIntentDataset(train_df["text"], train_labels, tokenizer)
    val_dataset = CollegeIntentDataset(val_df["text"], val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading pretrained {MODEL_NAME} with {len(unique_intents)} target classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_intents),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    training_history = []
    best_val_acc = 0.0

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_accuracy": round(val_acc, 4)
        }
        training_history.append(epoch_stats)
        print(f"Epoch {epoch}/{EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    elapsed_time = round(time.time() - start_time, 2)
    print(f"\nFine-tuning completed in {elapsed_time}s. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Saved fine-tuned BERT model to {OUTPUT_DIR}")

    with open(os.path.join(RESULTS_DIR, "bert_training_history.json"), "w") as f:
        json.dump({
            "history": training_history,
            "total_time_seconds": elapsed_time,
            "best_val_accuracy": best_val_acc,
            "device": str(device)
        }, f, indent=2)

if __name__ == "__main__":
    train_bert()
