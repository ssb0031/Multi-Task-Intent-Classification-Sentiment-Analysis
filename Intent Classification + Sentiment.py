import json
import random
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------
# Config (tuned for 16GB VRAM)
# ---------------------------------------------------------------------
DATA_PATH = "/kaggle/input/ds-intent-sentiment/augmented_intents.json"
MODEL_NAME = "xlm-roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 64                # shorter sequences -> less memory
BATCH_SIZE = 4              # per-step micro-batch
GRAD_ACC_STEPS = 4          # effective batch size = BATCH_SIZE * GRAD_ACC_STEPS
NUM_EPOCHS = 80             # (kept for compatibility; we use per-phase epochs below)
LR = 3e-5
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIPPING = 1.0

USE_AMP = True
USE_GRAD_CHECKPOINT = True

PHASE_SETTINGS = {
    1: {"frozen_backbone": True,  "bilstm": False, "fusion": False, "epochs": 15},
    2: {"frozen_backbone": False, "bilstm": False, "fusion": False, "epochs": 15},
    3: {"frozen_backbone": False, "bilstm": True,  "fusion": False, "epochs": 15},
    4: {"frozen_backbone": False, "bilstm": True,  "fusion": True,  "epochs": 15},
    5: {"frozen_backbone": False, "bilstm": True,  "fusion": True,  "epochs": 20}
}

# Create output directories
for i in range(1, 6):
    Path(f"runs/phase{i}").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------------------
print("Loading data...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

intents = data['intents']
all_patterns, all_tags, all_sentiments = [], [], []
tag_responses = defaultdict(list)

for intent in intents:
    for pattern in intent['patterns']:
        all_patterns.append(pattern)
        all_tags.append(intent['tag'])
        all_sentiments.append(intent['sentiment'])
    tag_responses[intent['tag']].extend(intent['responses'])

# Stable, deterministic tag list to align IDs <-> names
unique_tags = sorted(set(all_tags))
tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

sentiment_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}
idx_to_sentiment = {idx: s for s, idx in sentiment_to_idx.items()}

# Create indices and stratified split per tag—ensure both sets see each class
all_indices = list(range(len(all_patterns)))
train_indices, val_indices = [], []

tag_indices = defaultdict(list)
for idx, tag in enumerate(all_tags):
    tag_indices[tag].append(idx)

for tag, indices in tag_indices.items():
    n_samples = len(indices)
    if n_samples == 1:
        train_indices.extend(indices)
    elif n_samples == 2:
        train_indices.append(indices[0])
        val_indices.append(indices[1])
    else:
        indices_np = np.array(indices)
        dummy_labels = np.zeros(len(indices))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for tr, va in sss.split(indices_np, dummy_labels):
            train_indices.extend(indices_np[tr].tolist())
            val_indices.extend(indices_np[va].tolist())

# Verify coverage and fix missing tags in validation
train_tags = [all_tags[i] for i in train_indices]
val_tags = [all_tags[i] for i in val_indices]

print(f"Training set size: {len(train_indices)}")
print(f"Validation set size: {len(val_indices)}")
print(f"Unique tags in training: {len(set(train_tags))}/{len(unique_tags)}")
print(f"Unique tags in validation: {len(set(val_tags))}/{len(unique_tags)}")

missing_in_val = set(unique_tags) - set(val_tags)
if missing_in_val:
    print(f"Warning: These tags are missing in validation: {missing_in_val}")
    for tag in missing_in_val:
        for i in train_indices:
            if all_tags[i] == tag:
                train_indices.remove(i)
                val_indices.append(i)
                print(f"Moved sample {i} (tag: {tag}) from training to validation")
                break

train_tags = [all_tags[i] for i in train_indices]
val_tags = [all_tags[i] for i in val_indices]
print(f"After adjustment - Unique tags in training: {len(set(train_tags))}/{len(unique_tags)}")
print(f"After adjustment - Unique tags in validation: {len(set(val_tags))}/{len(unique_tags)}")

# Class weights (mild inverse-frequency)
tag_counts = Counter(all_tags)
class_weights = torch.tensor(
    [1 / np.log(1 + tag_counts[tag]) for tag in unique_tags],
    dtype=torch.float, device=DEVICE
)

# ---------------------------------------------------------------------
# Dataset / Dataloader
# ---------------------------------------------------------------------
class MentalHealthDataset(Dataset):
    def __init__(self, patterns, tags, sentiments, tokenizer, max_len):
        self.patterns = patterns
        self.tags = tags
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = str(self.patterns[idx])
        tag = self.tags[idx]
        sentiment = self.sentiments[idx]

        encoding = self.tokenizer.encode_plus(
            pattern,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'tag': torch.tensor(tag_to_idx[tag], dtype=torch.long),
            'sentiment': torch.tensor(sentiment_to_idx[sentiment], dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = MentalHealthDataset(
    [all_patterns[i] for i in train_indices],
    [all_tags[i] for i in train_indices],
    [all_sentiments[i] for i in train_indices],
    tokenizer, MAX_LEN
)
val_dataset = MentalHealthDataset(
    [all_patterns[i] for i in val_indices],
    [all_tags[i] for i in val_indices],
    [all_sentiments[i] for i in val_indices],
    tokenizer, MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class MultiTaskClassifier(nn.Module):
    def __init__(self, n_tags, n_sentiments, phase_config):
        super().__init__()
        self.phase_config = phase_config
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        # Allow grad checkpointing (requires use_cache=False)
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

        # Freeze backbone if needed
        if phase_config['frozen_backbone']:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        # BiLSTM for phase 3+
        self.bilstm = None
        if phase_config['bilstm']:
            self.bilstm = nn.LSTM(
                hidden_size, hidden_size // 2,
                num_layers=2, bidirectional=True,
                batch_first=True, dropout=0.3
            )

        # Sentiment embedding for fusion (phase 4+)
        self.sentiment_embedding = None
        self.fusion_layer = None
        if phase_config['fusion']:
            self.sentiment_embedding = nn.Embedding(n_sentiments, 64)
            # Add fusion layer to mix sentiment and semantic information
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_size + 64, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        intent_input_size = hidden_size
        if phase_config['fusion']:
            intent_input_size = hidden_size  # After fusion layer

        self.tag_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(intent_input_size, n_tags)
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, n_sentiments)
        )

    def forward(self, input_ids, attention_mask, sentiment_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, T, H]

        if self.bilstm is not None:
            sequence_output, _ = self.bilstm(sequence_output)

        cls_output = sequence_output[:, 0, :]  # [B, H]

        # Sentiment head
        sentiment_logits = self.sentiment_classifier(cls_output)

        # Intent head with optional sentiment fusion
        intent_features = cls_output
        if self.phase_config['fusion']:
            if sentiment_labels is not None:
                sent_emb = self.sentiment_embedding(sentiment_labels)
            else:
                pred_sent = torch.argmax(sentiment_logits, dim=1)
                sent_emb = self.sentiment_embedding(pred_sent)
            
            # Concatenate and apply fusion layer
            fused_features = torch.cat([cls_output, sent_emb], dim=1)
            intent_features = self.fusion_layer(fused_features)

        tag_logits = self.tag_classifier(intent_features)
        return tag_logits, sentiment_logits

n_tags = len(unique_tags)
n_sentiments = len(sentiment_to_idx)

# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------
tag_criterion = nn.CrossEntropyLoss(weight=class_weights)
sentiment_criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------
def calculate_ece(true, probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == true

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences >= bin_lower, confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece

# Fuzzy intent score prep
sbert = SentenceTransformer('all-MiniLM-L6-v2')
tag_descriptions = {}
for intent in intents:
    examples = " ".join(intent['patterns'][:3])
    tag_descriptions[intent['tag']] = f"{intent['tag']}: {examples}"
tag_embeddings = sbert.encode(list(tag_descriptions.values()))

def calculate_fuzzy_intent(true_tags, pred_tags):
    val_patterns = [all_patterns[i] for i in val_indices]
    pattern_embeddings = sbert.encode(val_patterns)
    similarities = np.dot(pattern_embeddings, tag_embeddings.T)
    pattern_norms = np.linalg.norm(pattern_embeddings, axis=1, keepdims=True)
    tag_norms = np.linalg.norm(tag_embeddings, axis=1, keepdims=True)
    similarities = similarities / (pattern_norms * tag_norms.T + 1e-12)

    correct = 0.0
    for i, (true, pred) in enumerate(zip(true_tags, pred_tags)):
        if true == pred:
            correct += 1.0
        else:
            pred_tag = idx_to_tag[pred]
            pred_tag_idx = unique_tags.index(pred_tag)
            if similarities[i, pred_tag_idx] >= 0.70:
                correct += 0.5
    return correct / len(true_tags)

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate(model, phase_config):
    model.eval()
    tag_preds, tag_true = [], []
    sentiment_preds, sentiment_true = [], []
    tag_probs, sentiment_probs = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            tags = batch['tag'].to(DEVICE)
            sentiments = batch['sentiment'].to(DEVICE)

            tag_logits, sentiment_logits = model(input_ids, attention_mask, None)
            # Fixed loss balancing (simpler approach)
            loss_tag = tag_criterion(tag_logits, tags)
            loss_sentiment = sentiment_criterion(sentiment_logits, sentiments)
            loss = loss_tag + 0.5 * loss_sentiment  # Fixed weight
            total_loss += loss.item()

            tag_probs.extend(torch.softmax(tag_logits, dim=1).cpu().numpy())
            sentiment_probs.extend(torch.softmax(sentiment_logits, dim=1).cpu().numpy())

            tag_preds.extend(torch.argmax(tag_logits, dim=1).cpu().numpy())
            sentiment_preds.extend(torch.argmax(sentiment_logits, dim=1).cpu().numpy())
            tag_true.extend(tags.cpu().numpy())
            sentiment_true.extend(sentiments.cpu().numpy())

    tag_accuracy = accuracy_score(tag_true, tag_preds)
    sentiment_accuracy = accuracy_score(sentiment_true, sentiment_preds)
    tag_f1_macro = f1_score(tag_true, tag_preds, average='macro')
    tag_f1_weighted = f1_score(tag_true, tag_preds, average='weighted')
    sentiment_f1_macro = f1_score(sentiment_true, sentiment_preds, average='macro')

    tag_ece = calculate_ece(tag_true, np.array(tag_probs))
    sentiment_ece = calculate_ece(sentiment_true, np.array(sentiment_probs))
    fuzzy_score = calculate_fuzzy_intent(tag_true, tag_preds)

    return {
        'total_loss': total_loss / len(val_loader),
        'tag_accuracy': tag_accuracy,
        'sentiment_accuracy': sentiment_accuracy,
        'tag_f1_macro': tag_f1_macro,
        'tag_f1_weighted': tag_f1_weighted,
        'sentiment_f1_macro': sentiment_f1_macro,
        'tag_ece': tag_ece,
        'sentiment_ece': sentiment_ece,
        'fuzzy_score': fuzzy_score,
        'tag_true': tag_true,
        'tag_preds': tag_preds,
        'sentiment_true': sentiment_true,
        'sentiment_preds': sentiment_preds,
        'tag_probs': tag_probs,
        'sentiment_probs': sentiment_probs
    }

# ---------------------------------------------------------------------
# Training (with AMP, grad accumulation, checkpointing, progressive unfreeze)
# ---------------------------------------------------------------------
from contextlib import nullcontext
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

def train_phase(phase_idx, model, optimizer, scheduler, start_epoch, end_epoch, progressive_unfreeze=False):
    best_val_loss = float('inf')
    patience_counter = 0
    total_loss = 0.0

    # Enable gradient checkpointing if available
    if USE_GRAD_CHECKPOINT and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    # Correct path for XLM-R layers
    encoder_layers = getattr(model.encoder, "layer", None)
    num_layers = len(encoder_layers) if encoder_layers is not None else 0

    for epoch in range(start_epoch, end_epoch):
        model.train()
        epoch_loss = 0.0

        # Progressive unfreezing (phase 2 only) - increased to 3 layers per 5 epochs
        if progressive_unfreeze and num_layers > 0:
            epoch_in_phase = epoch - start_epoch
            layers_to_unfreeze = min(3 * (epoch_in_phase // 5 + 1), num_layers)  # 3 layers every 5 epochs
            for i, layer in enumerate(encoder_layers):
                req_grad = (i >= num_layers - layers_to_unfreeze)
                for p in layer.parameters():
                    p.requires_grad = req_grad

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            tags = batch['tag'].to(DEVICE)
            sentiments = batch['sentiment'].to(DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                tag_logits, sentiment_logits = model(input_ids, attention_mask, sentiments)
                loss_tag = tag_criterion(tag_logits, tags)
                loss_sentiment = sentiment_criterion(sentiment_logits, sentiments)
                
                # Fixed loss balancing (simpler approach)
                loss = loss_tag + 0.5 * loss_sentiment
                loss = loss / GRAD_ACC_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACC_STEPS == 0:
                clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACC_STEPS  # undo normalization for logging

        avg_epoch_loss = epoch_loss / len(train_loader)
        total_loss += avg_epoch_loss

        # ---- Validation ----
        model.eval()
        val_metrics = evaluate(model, PHASE_SETTINGS[phase_idx + 1])
        val_loss = val_metrics['total_loss']
        scheduler.step(val_loss)

        print(f"Phase {phase_idx+1}, Epoch {epoch+1}/{end_epoch} | "
              f"Train {avg_epoch_loss:.4f} | Val {val_loss:.4f} | "
              f"TagAcc {val_metrics['tag_accuracy']:.4f} | "
              f"SentAcc {val_metrics['sentiment_accuracy']:.4f}")

        # Early stopping per phase
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"runs/phase{phase_idx+1}/best_model.pt")
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return total_loss / max(1, (end_epoch - start_epoch))

# ---------------------------------------------------------------------
# Training loop across phases
# ---------------------------------------------------------------------
history = defaultdict(list)
cumulative_epochs_before_phase = lambda p: sum(PHASE_SETTINGS[i]["epochs"] for i in range(1, p))

for phase in range(1, 6):
    print(f"\n==================== Starting Phase {phase} ====================")
    phase_config = PHASE_SETTINGS[phase]

    if phase == 1:
        model = MultiTaskClassifier(n_tags, n_sentiments, phase_config).to(DEVICE)
    else:
        # load the previous phase model weights
        prev_state = torch.load(f"runs/phase{phase-1}/best_model.pt", map_location=DEVICE)
        model = MultiTaskClassifier(n_tags, n_sentiments, phase_config).to(DEVICE)
        model.load_state_dict(prev_state, strict=False)  # allow new heads if added

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    progressive_unfreeze = (phase == 2)

    start_epoch = cumulative_epochs_before_phase(phase)
    end_epoch = cumulative_epochs_before_phase(phase + 1)

    avg_loss = train_phase(
        phase_idx=phase - 1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        progressive_unfreeze=progressive_unfreeze
    )

    # Evaluate after phase
    val_metrics = evaluate(model, phase_config)

    # Save history
    history['phase'].append(phase)
    history['train_loss'].append(avg_loss)
    history['val_loss'].append(val_metrics['total_loss'])
    history['tag_accuracy'].append(val_metrics['tag_accuracy'])
    history['sentiment_accuracy'].append(val_metrics['sentiment_accuracy'])
    history['fuzzy_score'].append(val_metrics['fuzzy_score'])
    history['tag_f1_macro'].append(val_metrics['tag_f1_macro'])
    history['tag_f1_weighted'].append(val_metrics['tag_f1_weighted'])
    history['sentiment_f1_macro'].append(val_metrics['sentiment_f1_macro'])
    history['tag_ece'].append(val_metrics['tag_ece'])
    history['sentiment_ece'].append(val_metrics['sentiment_ece'])

    # Metrics & plots per phase
    label_ids = list(range(n_tags))
    tag_names = [idx_to_tag[i] for i in label_ids]

    phase_metrics = {
        'tag_classification_report': classification_report(
            val_metrics['tag_true'],
            val_metrics['tag_preds'],
            labels=label_ids,
            target_names=tag_names,
            output_dict=True
        ),
        'sentiment_classification_report': classification_report(
            val_metrics['sentiment_true'],
            val_metrics['sentiment_preds'],
            labels=[0, 1, 2],
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        ),
        'confusion_matrices': {
            'tag': confusion_matrix(val_metrics['tag_true'], val_metrics['tag_preds'], labels=label_ids).tolist(),
            'sentiment': confusion_matrix(val_metrics['sentiment_true'], val_metrics['sentiment_preds'], labels=[0,1,2]).tolist()
        }
    }

    with open(f"runs/phase{phase}/metrics.json", "w") as f:
        json.dump(phase_metrics, f, indent=2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(phase_metrics['confusion_matrices']['tag'], annot=True, fmt='d')
    plt.title(f"Tag Confusion Matrix - Phase {phase}")
    plt.subplot(1, 2, 2)
    sns.heatmap(phase_metrics['confusion_matrices']['sentiment'], annot=True, fmt='d')
    plt.title(f"Sentiment Confusion Matrix - Phase {phase}")
    plt.tight_layout()
    plt.savefig(f"runs/phase{phase}/confusion_matrices.png")
    plt.close()

    # Save last model of the phase
    torch.save(model.state_dict(), f"runs/phase{phase}/last_model.pt")

    # Aggressive cleanup before next phase
    del optimizer, scheduler, model
    torch.cuda.empty_cache()
    import gc; gc.collect()

# ---------------------------------------------------------------------
# Training summary plots
# ---------------------------------------------------------------------
pd.DataFrame(history).to_csv("training_history.csv", index=False)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(history['phase'], history['train_loss'], 'o-', label='Train Loss')
plt.plot(history['phase'], history['val_loss'], 'o-', label='Val Loss')
plt.xlabel('Phase'); plt.ylabel('Loss'); plt.legend(); plt.title('Training and Validation Loss')

plt.subplot(2, 3, 2)
plt.plot(history['phase'], history['tag_accuracy'], 'o-', label='Tag Accuracy')
plt.plot(history['phase'], history['sentiment_accuracy'], 'o-', label='Sentiment Accuracy')
plt.xlabel('Phase'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Metrics')

plt.subplot(2, 3, 3)
plt.plot(history['phase'], history['fuzzy_score'], 'o-', label='Fuzzy Score')
plt.xlabel('Phase'); plt.ylabel('Score'); plt.legend(); plt.title('Fuzzy Intent Score')

plt.subplot(2, 3, 4)
plt.plot(history['phase'], history['tag_f1_macro'], 'o-', label='Macro F1')
plt.plot(history['phase'], history['tag_f1_weighted'], 'o-', label='Weighted F1')
plt.xlabel('Phase'); plt.ylabel('F1 Score'); plt.legend(); plt.title('Tag F1 Scores')

plt.subplot(2, 3, 5)
plt.plot(history['phase'], history['sentiment_f1_macro'], 'o-', label='Sentiment F1')
plt.xlabel('Phase'); plt.ylabel('F1 Score'); plt.legend(); plt.title('Sentiment F1 Score')

plt.subplot(2, 3, 6)
plt.plot(history['phase'], history['tag_ece'], 'o-', label='Tag ECE')
plt.plot(history['phase'], history['sentiment_ece'], 'o-', label='Sentiment ECE')
plt.xlabel('Phase'); plt.ylabel('ECE'); plt.legend(); plt.title('Calibration Error')

plt.tight_layout()
plt.savefig("training_metrics_summary.png")
plt.close()

# ---------------------------------------------------------------------
# Final evaluation with best model from Phase 5
# ---------------------------------------------------------------------
best_model = MultiTaskClassifier(n_tags, n_sentiments, PHASE_SETTINGS[5]).to(DEVICE)
if USE_GRAD_CHECKPOINT and hasattr(best_model.encoder, "gradient_checkpointing_enable"):
    best_model.encoder.gradient_checkpointing_enable()
if hasattr(best_model.encoder.config, "use_cache"):
    best_model.encoder.config.use_cache = False

state = torch.load("runs/phase5/best_model.pt", map_location=DEVICE)
best_model.load_state_dict(state)
final_metrics = evaluate(best_model, PHASE_SETTINGS[5])

label_ids = list(range(n_tags))
tag_names = [idx_to_tag[i] for i in label_ids]

print("Final Tag Classification Report:")
print(classification_report(
    final_metrics['tag_true'],
    final_metrics['tag_preds'],
    labels=label_ids,
    target_names=tag_names
))

print("Final Sentiment Classification Report:")
print(classification_report(
    final_metrics['sentiment_true'],
    final_metrics['sentiment_preds'],
    labels=[0,1,2],
    target_names=['negative','neutral','positive']
))

print(f"Final Fuzzy Intent Score: {final_metrics['fuzzy_score']:.4f}")
print(f"Tag ECE: {final_metrics['tag_ece']:.4f}")
print(f"Sentiment ECE: {final_metrics['sentiment_ece']:.4f}")

response_data = {
    'tag_responses': tag_responses,
    'val_indices': val_indices,
    'val_patterns': [all_patterns[i] for i in val_indices],
    'val_tags': [all_tags[i] for i in val_indices],
    'val_sentiments': [all_sentiments[i] for i in val_indices]
}
with open("response_data.json", "w") as f:
    json.dump(response_data, f, indent=2)

print("Training completed! Results saved to runs/ directories.")