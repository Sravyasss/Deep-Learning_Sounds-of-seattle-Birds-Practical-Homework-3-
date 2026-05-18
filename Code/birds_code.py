# %% [markdown]
# # DATA 5322 Practical Homework 3: Deep Learning
# 
# **Author:** Sravya sri Murala
# **Date:** May 17 2026
# 
# This notebook trains convolutional neural networks to classify bird species from mel-spectrogram representations of their calls. It covers data loading, exploratory analysis, binary classification, multi-class classification, and prediction on three external test clips.

# %%
# Inspecting the Data

import h5py
import numpy as np
from pathlib import Path

DATA_PATH = Path('../data/bird_spectrograms.hdf5')
print(f"File exists: {DATA_PATH.exists()}")
print(f"File size: {DATA_PATH.stat().st_size / (1024**2):.1f} MB\n")

with h5py.File(DATA_PATH, 'r') as f:
    print("HDF5 file structure")
    print("-" * 60)
    print("Top-level keys:", list(f.keys()))
    print()
    
    def inspect(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}")
            print(f"    shape: {obj.shape}, dtype: {obj.dtype}")
            if len(obj.attrs) > 0:
                print(f"    attributes: {dict(obj.attrs)}")
        elif isinstance(obj, h5py.Group):
            print(f"  Group: {name}")
    
    f.visititems(inspect)
    print()
    print("Top-level attributes:", dict(f.attrs))

# %%
# Imports necessary libraries

import os
import time
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# matplotlib defaults for good quality plots
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.figsize': (8, 5)
})
sns.set_palette("deep")

# Paths
PROJECT_ROOT = Path('..').resolve()
DATA_PATH = PROJECT_ROOT / 'data' / 'bird_spectrograms.hdf5'
FIG_DIR = PROJECT_ROOT / 'figures'
MODEL_DIR = PROJECT_ROOT / 'models'
FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Readable species names
SPECIES_NAMES = {
    'amecro': 'American Crow',
    'amerob': 'American Robin',
    'bewwre': "Bewick's Wren",
    'bkcchi': 'Black-capped Chickadee',
    'daejun': 'Dark-eyed Junco',
    'houfin': 'House Finch',
    'houspa': 'House Sparrow',
    'norfli': 'Northern Flicker',
    'rewbla': 'Red-winged Blackbird',
    'sonspa': 'Song Sparrow',
    'spotow': 'Spotted Towhee',
    'whcspa': 'White-crowned Sparrow'
}

print("TensorFlow:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))
print("Project root:", PROJECT_ROOT)

# %%
# Exploratory Data Analysis
# Class Distribution

with h5py.File(DATA_PATH, 'r') as f:
    species_codes = list(f.keys())
    sample_counts = {sp: f[sp].shape[-1] for sp in species_codes}

eda_df = pd.DataFrame({
    'Code': species_codes,
    'Species': [SPECIES_NAMES[s] for s in species_codes],
    'Samples': [sample_counts[s] for s in species_codes]
}).sort_values('Samples', ascending=False).reset_index(drop=True)

print(eda_df.to_string(index=False))
print(f"\nTotal samples: {eda_df['Samples'].sum()}")
print(f"Mean: {eda_df['Samples'].mean():.1f},  Median: {eda_df['Samples'].median():.0f}")
print(f"Min: {eda_df['Samples'].min()} ({eda_df.loc[eda_df['Samples'].idxmin(),'Species']}),  "
      f"Max: {eda_df['Samples'].max()} ({eda_df.loc[eda_df['Samples'].idxmax(),'Species']})")
print(f"Imbalance ratio: {eda_df['Samples'].max() / eda_df['Samples'].min():.1f}x")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette("viridis", len(eda_df))
bars = ax.bar(range(len(eda_df)), eda_df['Samples'], color=colors, edgecolor='black')
ax.set_xticks(range(len(eda_df)))
ax.set_xticklabels(eda_df['Species'], rotation=45, ha='right')
ax.set_ylabel('Number of Spectrogram Samples')
ax.set_xlabel('Bird Species')
ax.set_title('Class Distribution Across 12 Seattle Bird Species')
ax.axhline(eda_df['Samples'].mean(), color='red', linestyle='--', linewidth=1.5,
           label=f"Mean = {eda_df['Samples'].mean():.0f}")

for bar, count in zip(bars, eda_df['Samples']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            str(count), ha='center', fontsize=9)

ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / '01_class_distribution.png')
plt.show()

# %%
# Example Spectogram

fig, axes = plt.subplots(3, 4, figsize=(15, 9))
with h5py.File(DATA_PATH, 'r') as f:
    for ax, sp in zip(axes.flat, species_codes):
        
        # Take the first spectrogram 
        spec = f[sp][:, :, 0]
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(SPECIES_NAMES[sp], fontsize=10)
        ax.set_xlabel('Time frame')
        ax.set_ylabel('Frequency bin')

plt.suptitle('Representative Mel-Spectrogram for Each of the 12 Bird Species',
             y=1.00, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / '02_example_spectrograms.png')
plt.show()

# %%
# Data loading

def load_data(path, species_subset=None, verbose=True):
    """
    Load spectrograms from HDF5 with per-sample z-score normalization.
    Standardizing each spectrogram (mean 0, std 1) preserves the relative
    structure of features better than min-max scaling for audio data.
    """
    X_chunks, y_chunks = [], []
    with h5py.File(path, 'r') as f:
        species = species_subset if species_subset else list(f.keys())
        label_map = {sp: i for i, sp in enumerate(species)}
        for sp in species:
            
             # HDF5 stores 
            arr = f[sp][:]                      
            arr = np.transpose(arr, (2, 0, 1))   
            X_chunks.append(arr.astype(np.float32))
            y_chunks.append(np.full(arr.shape[0], label_map[sp], dtype=np.int64))
            if verbose:
                print(f"  Loaded {sp:>7} ({SPECIES_NAMES[sp]:>23}): {arr.shape[0]} samples")
    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    X = X[..., np.newaxis]
    
    # First attempt: per-sample min-max normalization to [0, 1].
    # This compressed the dynamic range too much and the models
    # could not learn from the spectrograms, so I switched to z-score below.
    # X_min = X.min(axis=(1, 2, 3), keepdims=True)
    # X_max = X.max(axis=(1, 2, 3), keepdims=True)
    # X = (X - X_min) / (X_max - X_min + 1e-8)
    
    #  per-sample z-score normalization
    mu = X.mean(axis=(1, 2, 3), keepdims=True)
    sd = X.std(axis=(1, 2, 3), keepdims=True) + 1e-8
    X = (X - mu) / sd
    if verbose:
        print(f"\nFinal X shape: {X.shape}, y shape: {y.shape}")
        print(f"After z-score: mean={X.mean():.3f}, std={X.std():.3f}, "
              f"min={X.min():.2f}, max={X.max():.2f}")
        print(f"Memory: {X.nbytes / 1024**3:.2f} GB (float32)")
    return X, y, label_map

# %%
# Loading all 12 species 
print("Loading full dataset (12 species)...")
t0 = time.time()
X_all, y_all, label_map = load_data(DATA_PATH, verbose=True)
print(f"\nLoad time: {time.time() - t0:.1f}s")

# Inverse map and class names for plotting
inv_label_map = {v: k for k, v in label_map.items()}
class_codes   = [inv_label_map[i] for i in range(len(label_map))]
class_names   = [SPECIES_NAMES[c] for c in class_codes]
n_classes     = len(class_names)

print(f"\nNumber of classes: {n_classes}")
print(f"Class names: {class_names}")

# %%
# Stratified split

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=SEED
)

# splitting the remaining 
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=SEED
)

print("Split sizes:")
print(f"  Train: {len(X_train):>5} ({len(X_train)/len(X_all):.1%})")
print(f"  Val:   {len(X_val):>5} ({len(X_val)/len(X_all):.1%})")
print(f"  Test:  {len(X_test):>5} ({len(X_test)/len(X_all):.1%})")

# Verifying
print("\nPer-class counts (train - val - test):")
train_counts = np.bincount(y_train, minlength=n_classes)
val_counts   = np.bincount(y_val,   minlength=n_classes)
test_counts  = np.bincount(y_test,  minlength=n_classes)
split_df = pd.DataFrame({
    'Species': class_names,
    'Train': train_counts, 'Val': val_counts, 'Test': test_counts
})
print(split_df.to_string(index=False))

# %%
# computing class weights 

class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(n_classes),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights_arr)}

cw_df = pd.DataFrame({
    'Species': class_names,
    'Train Count': train_counts,
    'Weight': [f'{w:.3f}' for w in class_weights_arr]
}).sort_values('Train Count', ascending=False)
print("Class weights for training:")
print(cw_df.to_string(index=False))

# %%
# Binary classification 

BINARY_SPECIES = ['houspa', 'spotow']   # 0 = House Sparrow, 1 = Spotted Towhee
BINARY_LABELS  = [SPECIES_NAMES[s] for s in BINARY_SPECIES]

# Loading the two species
X_bin, y_bin, bin_map = load_data(DATA_PATH, species_subset=BINARY_SPECIES, verbose=True)
print(f"\nBinary dataset shape: {X_bin.shape}")
print(f"Class counts: {dict(zip(BINARY_LABELS, np.bincount(y_bin)))}")

# Stratified split
Xb_tv, Xb_test, yb_tv, yb_test = train_test_split(
    X_bin, y_bin, test_size=0.15, stratify=y_bin, random_state=SEED
)
Xb_train, Xb_val, yb_train, yb_val = train_test_split(
    Xb_tv, yb_tv, test_size=0.1765, stratify=yb_tv, random_state=SEED
)

print(f"\nBinary splits:")
print(f"  Train: {len(Xb_train)}  (HouseSparrow={np.sum(yb_train==0)}, SpottedTowhee={np.sum(yb_train==1)})")
print(f"  Val:   {len(Xb_val)}    (HouseSparrow={np.sum(yb_val==0)}, SpottedTowhee={np.sum(yb_val==1)})")
print(f"  Test:  {len(Xb_test)}   (HouseSparrow={np.sum(yb_test==0)}, SpottedTowhee={np.sum(yb_test==1)})")

# Class weights for binary
bin_class_weights_arr = compute_class_weight(
    class_weight='balanced', classes=np.array([0,1]), y=yb_train)
bin_class_weights = {i: w for i, w in enumerate(bin_class_weights_arr)}
print(f"\nBinary class weights: {bin_class_weights}")

# %%
# Binary CNN architecture

def build_binary_cnn(input_shape, lr=1e-3):
    """
    Simple Conv2D network for binary spectrogram classification.
    Uses small filters, modest depth, global pooling, and moderate
    dropout for regularization on a relatively small dataset.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name='binary_cnn')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model

binary_model = build_binary_cnn(Xb_train.shape[1:])
binary_model.summary()
print(f"\nTotal parameters: {binary_model.count_params():,}")

# %%
# Training binary baseline

es_callback = callbacks.EarlyStopping(
    monitor='val_auc', mode='max', patience=10,
    restore_best_weights=True, verbose=1
)
rlr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

print("Training final binary model...")
t0 = time.time()
history_binary = binary_model.fit(
    Xb_train, yb_train,
    validation_data=(Xb_val, yb_val),
    epochs=40,
    batch_size=32,
    class_weight=bin_class_weights,
    callbacks=[es_callback, rlr_callback],
    verbose=1
)
binary_train_time = time.time() - t0
print(f"\nBinary training time: {binary_train_time:.1f}s ({binary_train_time/60:.1f} min)")

#  Quick check
val_probs = binary_model.predict(Xb_val, verbose=0).ravel()
val_preds = (val_probs > 0.5).astype(int)
print(f"\nValidation prediction distribution: {np.bincount(val_preds, minlength=2)}")
print(f"Validation true distribution:       {np.bincount(yb_val, minlength=2)}")
print(f"Best val_auc: {max(history_binary.history['val_auc']):.3f}")
print(f"Best val_accuracy: {max(history_binary.history['val_accuracy']):.3f}")

# %%
# Evaluting the Binary Model
from sklearn.metrics import roc_auc_score
test_probs_bin = binary_model.predict(Xb_test, verbose=0).ravel()
test_preds_bin = (test_probs_bin > 0.5).astype(int)

# Classification report 
print("BINARY MODEL — TEST SET METRICS")
print("-" * 60)
print(classification_report(yb_test, test_preds_bin,
                            target_names=BINARY_LABELS, digits=3))

# Aggregating metrics
test_acc = accuracy_score(yb_test, test_preds_bin)
test_auc = roc_auc_score(yb_test, test_probs_bin)
test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
    yb_test, test_preds_bin, average='binary', pos_label=1, zero_division=0)

print(f"\nTest Accuracy:  {test_acc:.3f}")
print(f"Test AUC:       {test_auc:.3f}")
print(f"Test Precision: {test_prec:.3f}")
print(f"Test Recall:    {test_rec:.3f}")
print(f"Test F1:        {test_f1:.3f}")

# %%
# Diagnostic Plots of the binary model

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
hist = history_binary.history
epochs_ran = range(1, len(hist['loss']) + 1)

axes[0].plot(epochs_ran, hist['loss'], label='Training', linewidth=2)
axes[0].plot(epochs_ran, hist['val_loss'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Binary Cross-Entropy Loss')
axes[0].set_title('Binary Model: Loss Curves'); axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_ran, hist['auc'], label='Training AUC', linewidth=2)
axes[1].plot(epochs_ran, hist['val_auc'], label='Validation AUC', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC')
axes[1].set_title('Binary Model: AUC Curves'); axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / '03_binary_training_curves.png')
plt.show()

# Confusion matrix
cm_bin = confusion_matrix(yb_test, test_preds_bin)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues',
            xticklabels=BINARY_LABELS, yticklabels=BINARY_LABELS,
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_xlabel('Predicted Species'); ax.set_ylabel('True Species')
ax.set_title('Binary Model: Test Set Confusion Matrix')
plt.tight_layout()
plt.savefig(FIG_DIR / '04_binary_confusion_matrix.png')
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(yb_test, test_probs_bin)
roc_auc_value = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='steelblue', lw=2.5,
        label=f'CNN (AUC = {roc_auc_value:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier (AUC = 0.5)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title(f'Binary Model: ROC Curve\n({BINARY_LABELS[0]} vs {BINARY_LABELS[1]})')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(FIG_DIR / '05_binary_roc_curve.png')
plt.show()

# 4. Probability distribution by true class
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(test_probs_bin[yb_test == 0], bins=20, alpha=0.6,
        label=f'True {BINARY_LABELS[0]}', color='steelblue', edgecolor='black')
ax.hist(test_probs_bin[yb_test == 1], bins=20, alpha=0.6,
        label=f'True {BINARY_LABELS[1]}', color='darkorange', edgecolor='black')
ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Decision threshold (0.5)')
ax.set_xlabel('Predicted probability of Spotted Towhee')
ax.set_ylabel('Number of test samples')
ax.set_title('Binary Model: Predicted Probability Distribution by True Class')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / '06_binary_probability_distribution.png')
plt.show()

# %%
# Code to Save binary model
binary_model.save(MODEL_DIR / 'binary_cnn.keras')

binary_results = {
    'pair': BINARY_LABELS,
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'test_precision': float(test_prec),
    'test_recall': float(test_rec),
    'test_f1': float(test_f1),
    'training_time_seconds': float(binary_train_time),
    'n_train': int(len(Xb_train)),
    'n_val': int(len(Xb_val)),
    'n_test': int(len(Xb_test)),
    'parameters': int(binary_model.count_params()),
    'best_val_auc': float(max(history_binary.history['val_auc'])),
    'best_val_accuracy': float(max(history_binary.history['val_accuracy']))
}

import json
with open(MODEL_DIR / 'binary_results.json', 'w') as f:
    json.dump(binary_results, f, indent=2)

print("Binary model saved.")
print(json.dumps(binary_results, indent=2))

# %%
# Reloading all 12 species with z-score normalization

print("Reloading full dataset with z-score normalization...")
t0 = time.time()
X_all, y_all, label_map = load_data(DATA_PATH, verbose=True)
print(f"Load time: {time.time() - t0:.1f}s")

# Recomputing the splits
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=SEED
)

inv_label_map = {v: k for k, v in label_map.items()}
class_codes   = [inv_label_map[i] for i in range(len(label_map))]
class_names   = [SPECIES_NAMES[c] for c in class_codes]
n_classes     = len(class_names)

# Class weights
class_weights_arr = compute_class_weight(
    class_weight='balanced', classes=np.arange(n_classes), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights_arr)}

print(f"\nSplits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
print(f"Number of classes: {n_classes}")

# %%
# Building Multi-class CNN architecture

def build_multiclass_cnn(input_shape, n_classes,
                        base_filters=16,
                        n_conv_blocks=3,
                        dropout=0.4,
                        dense_units=128,
                        lr=1e-3,
                        l2_reg=1e-4):
    """
    Conv2D network for multi-class bird call classification.
    Mirrors the binary architecture but with softmax output and
    sparse categorical crossentropy loss for 12-class prediction.
    """
    model = models.Sequential(name='multiclass_cnn')
    model.add(layers.Input(shape=input_shape))
    
    f = base_filters
    for i in range(n_conv_blocks):
        reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
        model.add(layers.Conv2D(f, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=reg, name=f'conv{i+1}a'))
        model.add(layers.Conv2D(f, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=reg, name=f'conv{i+1}b'))
        model.add(layers.MaxPooling2D((2, 2), name=f'pool{i+1}'))
        f *= 2
    
    model.add(layers.GlobalAveragePooling2D(name='gap'))
    model.add(layers.Dense(dense_units, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg), name='dense'))
    model.add(layers.Dropout(dropout, name='drop'))
    model.add(layers.Dense(n_classes, activation='softmax', name='output'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    return model

demo_multi = build_multiclass_cnn(X_train.shape[1:], n_classes)
demo_multi.summary()
print(f"\nTotal parameters: {demo_multi.count_params():,}")

# %%
# Training multi-class

multi_model = build_multiclass_cnn(X_train.shape[1:], n_classes)

es_callback = callbacks.EarlyStopping(
    monitor='val_accuracy', mode='max', patience=12,
    restore_best_weights=True, verbose=1
)
rlr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

print("Training multi-class baseline...")
print(f"  Training samples: {len(X_train)}, Classes: {n_classes}")
t0 = time.time()
history_multi_baseline = multi_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[es_callback, rlr_callback],
    verbose=1
)
multi_baseline_time = time.time() - t0
print(f"\nTraining time: {multi_baseline_time:.1f}s ({multi_baseline_time/60:.1f} min)")

# Quick checking
val_probs_multi = multi_model.predict(X_val, verbose=0)
val_preds_multi = np.argmax(val_probs_multi, axis=1)
print(f"\nValidation prediction distribution across 12 classes:")
print(np.bincount(val_preds_multi, minlength=n_classes))
print(f"\nValidation true distribution across 12 classes:")
print(np.bincount(y_val, minlength=n_classes))
print(f"\nBest val_accuracy: {max(history_multi_baseline.history['val_accuracy']):.3f}")
print(f"Best val_top3_acc: {max(history_multi_baseline.history['val_top3_acc']):.3f}")

# %%
# Building a balanced 12-class tf.data.Dataset

def make_balanced_multiclass_dataset(X, y, n_classes, batch_size=32, shuffle_buffer=200):
    """
    Create a tf.data.Dataset that samples uniformly across classes.
    Each class becomes its own infinite dataset, and we sample equally
    from each. This prevents majority-class collapse during training.
    """
    per_class_datasets = []
    for c in range(n_classes):
        mask = (y == c)
        Xc, yc = X[mask], y[mask]
        ds = (tf.data.Dataset.from_tensor_slices((Xc, yc))
              .shuffle(shuffle_buffer, seed=SEED + c)
              .repeat())
        per_class_datasets.append(ds)
    
    # Equal sample of all 12 classes
    weights = [1.0 / n_classes] * n_classes
    balanced_ds = tf.data.Dataset.sample_from_datasets(
        per_class_datasets, weights=weights, seed=SEED
    )
    balanced_ds = balanced_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return balanced_ds

# Building balanced training dataset
train_ds_multi = make_balanced_multiclass_dataset(
    X_train, y_train, n_classes, batch_size=32
)

val_ds_multi = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .batch(32).prefetch(tf.data.AUTOTUNE))

# Steps per epoch
steps_per_epoch_multi = len(X_train) // 32
print(f"Steps per epoch: {steps_per_epoch_multi}")
print(f"Each batch will have ~{32/n_classes:.1f} samples per class on average")

# %%
# Training multi-class model with balanced batch sampling

multi_model_balanced = build_multiclass_cnn(X_train.shape[1:], n_classes)

es_callback = callbacks.EarlyStopping(
    monitor='val_accuracy', mode='max', patience=15,
    restore_best_weights=True, verbose=1
)
rlr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1
)

print("Training multi-class model with balanced batch sampling...")
t0 = time.time()
history_multi = multi_model_balanced.fit(
    train_ds_multi,
    validation_data=val_ds_multi,
    epochs=60,
    steps_per_epoch=steps_per_epoch_multi,
    callbacks=[es_callback, rlr_callback],
    verbose=1
)
multi_train_time = time.time() - t0
print(f"\nTraining time: {multi_train_time:.1f}s ({multi_train_time/60:.1f} min)")

val_probs_multi = multi_model_balanced.predict(X_val, verbose=0)
val_preds_multi = np.argmax(val_probs_multi, axis=1)
print(f"\nPrediction distribution: {np.bincount(val_preds_multi, minlength=n_classes)}")
print(f"True distribution:       {np.bincount(y_val, minlength=n_classes)}")
print(f"\nBest val_accuracy: {max(history_multi.history['val_accuracy']):.3f}")
print(f"Best val_top3_acc: {max(history_multi.history['val_top3_acc']):.3f}")
print(f"Majority-class baseline: {np.bincount(y_val).max() / len(y_val):.3f}")
print(f"Random baseline: {1/n_classes:.3f}")

# Accuracy Per class
print("\nPer-class accuracy:")
for c in range(n_classes):
    mask = (y_val == c)
    if mask.sum() > 0:
        acc = np.mean(val_preds_multi[mask] == c)
        print(f"  {class_names[c]:>25}: {acc:.2%} ({int(mask.sum())} samples)")

# %%
# Multi-class diagnostic to see if the model actually learning?

val_probs_multi = multi_model.predict(X_val, verbose=0)
val_preds_multi = np.argmax(val_probs_multi, axis=1)

print("Prediction distribution:", np.bincount(val_preds_multi, minlength=n_classes))
print("True distribution:      ", np.bincount(y_val, minlength=n_classes))
print()
print(f"Validation accuracy: {np.mean(val_preds_multi == y_val):.3f}")
print(f"Majority-class accuracy: {np.bincount(y_val).max() / len(y_val):.3f}")
print()

# Accuracy per-class
print("Per-class accuracy:")
for c in range(n_classes):
    mask = (y_val == c)
    if mask.sum() > 0:
        acc = np.mean(val_preds_multi[mask] == c)
        print(f"  {class_names[c]:>25}: {acc:.2%} ({int(mask.sum())} samples)")

hs_idx = class_codes.index('houspa')
hs_predicted_mask = (val_preds_multi == hs_idx)
hs_true_mask = (y_val == hs_idx)
print(f"\nHouse Sparrow analysis:")
print(f"  Predicted as HS: {hs_predicted_mask.sum()}")
print(f"  Actually HS:     {hs_true_mask.sum()}")
print(f"  Correctly predicted HS: {(hs_predicted_mask & hs_true_mask).sum()}")
print(f"  If model predicted all-HS: would get {hs_true_mask.sum()} right (acc = {hs_true_mask.mean():.3f})")

# %%
# Multiclass with No class weights 

def build_multiclass_cnn_v2(input_shape, n_classes, lr=5e-4):
    """Slightly larger, simpler CNN for 12-class classification."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ], name='multiclass_cnn_v2')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    return model

multi_model_v2 = build_multiclass_cnn_v2(X_train.shape[1:], n_classes)
print(f"Parameters: {multi_model_v2.count_params():,}")

es_callback = callbacks.EarlyStopping(
    monitor='val_accuracy', mode='max', patience=12,
    restore_best_weights=True, verbose=1
)
rlr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

print("\nTraining multi-class v2 (no class weights, larger model)...")
t0 = time.time()
history_multi_v2 = multi_model_v2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[es_callback, rlr_callback],
    verbose=1
)
multi_v2_time = time.time() - t0
print(f"\nTraining time: {multi_v2_time:.1f}s ({multi_v2_time/60:.1f} min)")

val_probs_v2 = multi_model_v2.predict(X_val, verbose=0)
val_preds_v2 = np.argmax(val_probs_v2, axis=1)
print(f"\nPrediction distribution: {np.bincount(val_preds_v2, minlength=n_classes)}")
print(f"True distribution:       {np.bincount(y_val, minlength=n_classes)}")
print(f"\nBest val_accuracy: {max(history_multi_v2.history['val_accuracy']):.3f}")
print(f"Best val_top3_acc: {max(history_multi_v2.history['val_top3_acc']):.3f}")
print(f"Majority baseline: {np.bincount(y_val).max() / len(y_val):.3f}")
print(f"Random baseline:   {1/n_classes:.3f}")

print("\nPer-class accuracy:")
for c in range(n_classes):
    mask = (y_val == c)
    if mask.sum() > 0:
        acc = np.mean(val_preds_v2[mask] == c)
        print(f"  {class_names[c]:>25}: {acc:.2%} ({int(mask.sum())} samples)")

# %%
# Multi-class test set Evaluation

test_probs = multi_model_v2.predict(X_test, verbose=0)
test_preds = np.argmax(test_probs, axis=1)

# Accuracy overall
test_acc_multi = accuracy_score(y_test, test_preds)
test_top3_acc = np.mean([y_test[i] in np.argsort(test_probs[i])[-3:] 
                         for i in range(len(y_test))])

print("MULTI-CLASS MODEL — TEST SET METRICS")
print("-" * 60)
print(f"\nTest Accuracy:    {test_acc_multi:.3f}")
print(f"Test Top-3 Acc:   {test_top3_acc:.3f}")
print(f"Random baseline:  {1/n_classes:.3f}")
print(f"Majority baseline: {np.bincount(y_test).max() / len(y_test):.3f}\n")

# classification report
print("Per-class classification report:")
print(classification_report(y_test, test_preds, target_names=class_names, 
                            digits=3, zero_division=0))

# Macro-averaged metrics
prec, rec, f1, support = precision_recall_fscore_support(
    y_test, test_preds, average=None, zero_division=0)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, test_preds, average='macro', zero_division=0)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_test, test_preds, average='weighted', zero_division=0)

print(f"\nMacro-averaged    Precision: {prec_macro:.3f}, "
      f"Recall: {rec_macro:.3f}, F1: {f1_macro:.3f}")
print(f"Weighted-averaged Precision: {prec_weighted:.3f}, "
      f"Recall: {rec_weighted:.3f}, F1: {f1_weighted:.3f}")

# %%
# Multi-class diagnostic plots

# 1. Training curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
hist = history_multi_v2.history
epochs_ran = range(1, len(hist['loss']) + 1)

axes[0].plot(epochs_ran, hist['loss'], label='Training', linewidth=2)
axes[0].plot(epochs_ran, hist['val_loss'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Sparse Categorical Cross-Entropy Loss')
axes[0].set_title('Multi-class Model: Loss Curves'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_ran, hist['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(epochs_ran, hist['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[1].plot(epochs_ran, hist['val_top3_acc'], label='Val Top-3 Accuracy',
             linewidth=2, linestyle='--')
axes[1].axhline(1/n_classes, color='gray', linestyle=':', linewidth=1,
                label=f'Random baseline (1/{n_classes})')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Multi-class Model: Accuracy Curves'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / '07_multiclass_training_curves.png')
plt.show()

# Normalized Confusion matrix 
cm = confusion_matrix(y_test, test_preds)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion of true class'}, ax=ax,
            annot_kws={'fontsize': 8})
ax.set_xlabel('Predicted Species'); ax.set_ylabel('True Species')
ax.set_title('Multi-class Model: Normalized Confusion Matrix (Test Set)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / '08_multiclass_confusion_matrix.png')
plt.show()

# Precision/recall/F1 bar chart
metrics_df = pd.DataFrame({
    'Species': class_names,
    'Precision': prec,
    'Recall': rec,
    'F1': f1,
    'Test samples': support
}).sort_values('F1', ascending=True)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(metrics_df))
width = 0.27
ax.barh(x - width, metrics_df['Precision'], width, label='Precision', color='steelblue')
ax.barh(x,         metrics_df['Recall'],    width, label='Recall',    color='darkorange')
ax.barh(x + width, metrics_df['F1'],        width, label='F1',        color='forestgreen')
ax.set_yticks(x)
ax.set_yticklabels([f"{name} (n={n})" for name, n in zip(metrics_df['Species'],
                                                          metrics_df['Test samples'])])
ax.set_xlabel('Score')
ax.set_title('Multi-class Model: Per-Species Test Performance')
ax.set_xlim(0, 1.0)
ax.legend(loc='lower right'); ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(FIG_DIR / '09_multiclass_per_class_metrics.png')
plt.show()

# Multi-class ROC curves
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

fig, ax = plt.subplots(figsize=(8, 7))
for i in range(n_classes):
    if y_test_bin[:, i].sum() > 0:  # only if the class has test samples
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], test_probs[:, i])
        auc_i = auc(fpr_i, tpr_i)
        ax.plot(fpr_i, tpr_i, lw=1.5, label=f'{class_names[i]} (AUC={auc_i:.2f})')

ax.plot([0,1], [0,1], 'k--', lw=0.8, alpha=0.5)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Multi-class Model: One-vs-Rest ROC Curves (Test Set)')
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / '10_multiclass_roc_curves.png')
plt.show()

# %%
# Saving multi-class model and the record results

multi_model_v2.save(MODEL_DIR / 'multiclass_cnn.keras')

multiclass_results = {
    'n_classes': int(n_classes),
    'classes': class_names,
    'test_accuracy': float(test_acc_multi),
    'test_top3_accuracy': float(test_top3_acc),
    'macro_precision': float(prec_macro),
    'macro_recall': float(rec_macro),
    'macro_f1': float(f1_macro),
    'weighted_f1': float(f1_weighted),
    'random_baseline': float(1/n_classes),
    'majority_baseline': float(np.bincount(y_test).max() / len(y_test)),
    'training_time_seconds': float(multi_v2_time),
    'n_train': int(len(X_train)),
    'n_val': int(len(X_val)),
    'n_test': int(len(X_test)),
    'parameters': int(multi_model_v2.count_params()),
    'best_val_accuracy': float(max(history_multi_v2.history['val_accuracy'])),
    'best_val_top3_acc': float(max(history_multi_v2.history['val_top3_acc']))
}

with open(MODEL_DIR / 'multiclass_results.json', 'w') as f:
    json.dump(multiclass_results, f, indent=2)

print("Multi-class model saved.")
print(json.dumps(multiclass_results, indent=2))

# %%
# Verifying the clips

TEST_CLIPS_DIR = PROJECT_ROOT / 'data' / 'test_birds'
print(f"Test clips directory: {TEST_CLIPS_DIR}")
print(f"Exists: {TEST_CLIPS_DIR.exists()}")
print(f"\nContents:")
if TEST_CLIPS_DIR.exists():
    for f in sorted(TEST_CLIPS_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")

# %%
# Inspecting the clips 

import librosa
import librosa.display

TEST_CLIPS_DIR = PROJECT_ROOT / 'data' / 'test_birds'
test_clip_paths = sorted(TEST_CLIPS_DIR.glob('*.mp3'))

print("Test clip properties")
print("-" * 60)
for path in test_clip_paths:
    y_raw, sr = librosa.load(path, sr=22050)
    duration = len(y_raw) / sr
    print(f"\n{path.name}:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Number of samples: {len(y_raw)}")
    print(f"  Audio range: [{y_raw.min():.3f}, {y_raw.max():.3f}]")
    print(f"  RMS energy: {np.sqrt((y_raw**2).mean()):.4f}")

# %%
# Processing the clips and predicting the species

import librosa

def mp3_to_segments(filepath, sr=22050, segment_sec=2.0, hop_sec=1.0,
                    n_mels=128, n_time=517):
    """
    Slice an mp3 into overlapping 2-second segments and convert each
    to a 128 x 517 mel-spectrogram matching the training data format.
    """
    y, sr = librosa.load(filepath, sr=sr)
    seg_len = int(segment_sec * sr)
    hop_len_seg = int(hop_sec * sr)
    
    # mel-spectrogram parameters
    n_fft = 2048
    hop_length = max(1, seg_len // n_time)
    
    segments, start_times = [], []
    
    if len(y) < seg_len:
        y_padded = np.pad(y, (0, seg_len - len(y)))
        S = librosa.feature.melspectrogram(
            y=y_padded, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        if S_db.shape[1] < n_time:
            S_db = np.pad(S_db, ((0,0),(0, n_time - S_db.shape[1])))
        else:
            S_db = S_db[:, :n_time]
        segments.append(S_db)
        start_times.append(0.0)
    else:
        for start in range(0, len(y) - seg_len + 1, hop_len_seg):
            segment = y[start:start + seg_len]
            S = librosa.feature.melspectrogram(
                y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            if S_db.shape[1] < n_time:
                S_db = np.pad(S_db, ((0,0),(0, n_time - S_db.shape[1])))
            else:
                S_db = S_db[:, :n_time]
            segments.append(S_db)
            start_times.append(start / sr)
    
    X_segs = np.array(segments).astype(np.float32)
    # Applying the same z-score normalization per each segment
    mu = X_segs.mean(axis=(1, 2), keepdims=True)
    sd = X_segs.std(axis=(1, 2), keepdims=True) + 1e-8
    X_segs = (X_segs - mu) / sd
    X_segs = X_segs[..., np.newaxis]
    return X_segs, np.array(start_times), y, sr


# Processing all 3 clips
test_clip_paths = sorted(TEST_CLIPS_DIR.glob('*.mp3'))
clip_results = []

for clip_path in test_clip_paths:
    print(f"\nProcessing {clip_path.name}...")
    X_segs, start_times, raw_audio, sr = mp3_to_segments(clip_path)
    print(f"  Generated {len(X_segs)} segments of shape {X_segs.shape[1:]}")
    
    # Predicting with the multi-class model
    probs = multi_model_v2.predict(X_segs, verbose=0) 
    
    # top prediction per each segment
    seg_top1 = np.argmax(probs, axis=1)
    seg_top1_names = [class_names[i] for i in seg_top1]
    
    # Mean probability aggregation
    avg_probs = probs.mean(axis=0)
    top3_idx = np.argsort(avg_probs)[::-1][:3]
    
    pred_variance = probs.std(axis=0).mean()

    unique_segment_species = set(seg_top1_names)
    
    clip_results.append({
        'clip': clip_path.name,
        'duration_sec': len(raw_audio) / sr,
        'n_segments': len(X_segs),
        'avg_probs': avg_probs,
        'per_segment_probs': probs,
        'per_segment_top1': seg_top1_names,
        'start_times': start_times,
        'top3_idx': top3_idx,
        'top3_names': [class_names[i] for i in top3_idx],
        'top3_confs': [avg_probs[i] for i in top3_idx],
        'pred_variance': pred_variance,
        'unique_species_in_segments': unique_segment_species,
        'raw_audio': raw_audio,
        'sr': sr
    })
    
    print(f"  Top-3 (averaged): {[(n, f'{c:.2%}') for n, c in zip([class_names[i] for i in top3_idx], [avg_probs[i] for i in top3_idx])]}")
    print(f"  Unique top-1 species across segments: {unique_segment_species}")
    print(f"  Per-segment prediction variance: {pred_variance:.3f}")

# summary table
summary_rows = []
for r in clip_results:
    summary_rows.append({
        'Clip': r['clip'],
        'Duration (s)': f"{r['duration_sec']:.1f}",
        'Top-1 Prediction': r['top3_names'][0],
        'Top-1 Confidence': f"{r['top3_confs'][0]:.1%}",
        'Top-2 Prediction': r['top3_names'][1],
        'Top-2 Confidence': f"{r['top3_confs'][1]:.1%}",
        'Top-3 Prediction': r['top3_names'][2],
        'Top-3 Confidence': f"{r['top3_confs'][2]:.1%}",
        '# Distinct Top-1 Across Segments': len(r['unique_species_in_segments']),
        'Likely Multi-bird?': 'Yes' if len(r['unique_species_in_segments']) > 1 else 'No'
    })

summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 80)
print("MYSTERY CLIP PREDICTIONS SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False))

# Saving the summary 
summary_df.to_csv(FIG_DIR / 'mystery_clip_predictions.csv', index=False)

# %%
# Probability heatmap for each test clip

fig, axes = plt.subplots(3, 1, figsize=(13, 12))

for ax, r in zip(axes, clip_results):
    probs = r['per_segment_probs']  
    
    im = ax.imshow(probs.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names)
    
    n_segs = len(r['start_times'])
    n_ticks = min(n_segs, 12)
    tick_idx = np.linspace(0, n_segs - 1, n_ticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{r['start_times'][i]:.1f}s" for i in tick_idx])
    ax.set_xlabel('Segment start time (seconds)')
    
    n_unique = len(r['unique_species_in_segments'])
    ax.set_title(f"{r['clip']}  —  {n_segs} segments, "
                 f"{n_unique} distinct top-1 species "
                 f"({'multi-bird candidate' if n_unique > 1 else 'single-bird'})")
    
    plt.colorbar(im, ax=ax, label='Predicted probability')

plt.tight_layout()
plt.savefig(FIG_DIR / '11_mystery_clip_heatmaps.png')
plt.show()

# %%
# Full-clip spectrogram for each test clip

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for i, r in enumerate(clip_results):
    y_audio = r['raw_audio']
    sr = r['sr']
    t = np.arange(len(y_audio)) / sr
    
    # Waveform
    axes[i, 0].plot(t, y_audio, color='steelblue', linewidth=0.5)
    axes[i, 0].set_xlim(0, t.max())
    axes[i, 0].set_xlabel('Time (s)')
    axes[i, 0].set_ylabel('Amplitude')
    axes[i, 0].set_title(f"{r['clip']} — Waveform")
    axes[i, 0].grid(alpha=0.3)
    
    # Full spectrogram 
    S = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, n_fft=2048)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = axes[i, 1].imshow(S_db, aspect='auto', origin='lower', cmap='magma',
                            extent=[0, len(y_audio)/sr, 0, 128])
    axes[i, 1].set_xlabel('Time (s)')
    axes[i, 1].set_ylabel('Mel frequency bin')
    axes[i, 1].set_title(f"{r['clip']} — Mel-Spectrogram")

plt.tight_layout()
plt.savefig(FIG_DIR / '12_mystery_clip_waveforms_specs.png')
plt.show()

# %%
# Architecture comparison table 

arch_comparison = pd.DataFrame([
    {
        'Model': 'Binary CNN (final)',
        'Task': 'Binary (Spotted Towhee vs House Sparrow)',
        'Architecture': '3 conv blocks (16/32/64 filters) + GAP + Dense(32)',
        'Parameters': 25409,
        'Test Accuracy': 0.784,
        'Macro F1': 0.598,
        'Top-1 Confidence': 'N/A',
        'Training Time (s)': 17.4
    },
    {
        'Model': 'Multi-class baseline (collapsed)',
        'Task': '12-class with class weights',
        'Architecture': '3 conv blocks (16/32/64) + GAP + Dense(128)',
        'Parameters': 81660,
        'Test Accuracy': 0.322,  # majority-class fallback
        'Macro F1': 0.040,
        'Top-1 Confidence': 'N/A',
        'Training Time (s)': 'collapsed'
    },
    {
        'Model': 'Multi-class balanced sampling',
        'Task': '12-class with balanced batches',
        'Architecture': '3 conv blocks (16/32/64) + GAP + Dense(128)',
        'Parameters': 81660,
        'Test Accuracy': 0.144,  # overcorrected
        'Macro F1': 0.085,
        'Top-1 Confidence': 'N/A',
        'Training Time (s)': 194.6
    },
    {
        'Model': 'Multi-class CNN (final)',
        'Task': '12-class no class weights, larger model',
        'Architecture': '4 conv blocks (32/64/128/128) + GAP + Dense(128)',
        'Parameters': 258316,
        'Test Accuracy': 0.500,
        'Macro F1': 0.299,
        'Top-1 Confidence': '72.8% top-3',
        'Training Time (s)': 289.0
    }
])

print("Architecture comparison:")
print(arch_comparison.to_string(index=False))
arch_comparison.to_csv(FIG_DIR / 'architecture_comparison.csv', index=False)

# %%
# Final summary

print("PROJECT SUMMARY")
print("-" * 70)
print(f"\nBinary Classification (Spotted Towhee vs House Sparrow):")
print(f"  Test Accuracy: {binary_results['test_accuracy']:.3f}")
print(f"  Test AUC:      {binary_results['test_auc']:.3f}")
print(f"  Test Macro F1: 0.598")
print(f"  Train Time:    {binary_results['training_time_seconds']:.1f} s")

print(f"\nMulti-class Classification (12 species):")
print(f"  Test Accuracy:    {multiclass_results['test_accuracy']:.3f}")
print(f"  Test Top-3 Acc:   {multiclass_results['test_top3_accuracy']:.3f}")
print(f"  Macro F1:         {multiclass_results['macro_f1']:.3f}")
print(f"  Weighted F1:      {multiclass_results['weighted_f1']:.3f}")
print(f"  Train Time:       {multiclass_results['training_time_seconds']:.1f} s")

print(f"\nMystery Clip Predictions:")
for r in clip_results:
    print(f"  {r['clip']:>12} ({r['duration_sec']:.1f}s): "
          f"{r['top3_names'][0]} ({r['top3_confs'][0]:.1%}); "
          f"{len(r['unique_species_in_segments'])} distinct top-1 across segments")

print(f"\nAll figures saved to: {FIG_DIR}")
print(f"All models saved to:  {MODEL_DIR}")

# %%
# Model comparison plot 

comparison_data = pd.DataFrame([
    {'Model': 'Multi-class\nBaseline\n(class weights)', 
     'Test Acc': 0.322, 'Macro F1': 0.040, 'Status': 'Collapsed to\nmajority'},
    {'Model': 'Multi-class\nBalanced\nSampling', 
     'Test Acc': 0.144, 'Macro F1': 0.085, 'Status': 'Over-corrected'},
    {'Model': 'Multi-class\nFinal\n(larger model)', 
     'Test Acc': 0.500, 'Macro F1': 0.299, 'Status': 'Working\nmodel'},
])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Accuracy comparison 
x = np.arange(len(comparison_data))
colors = ['#d62728', '#ff7f0e', '#2ca02c']
axes[0].bar(x, comparison_data['Test Acc'], color=colors, edgecolor='black')
axes[0].axhline(1/12, color='gray', linestyle=':', linewidth=1.5,
                label=f'Random baseline (1/12 = {1/12:.3f})')
axes[0].axhline(0.319, color='gray', linestyle='--', linewidth=1.5,
                label='Majority-class baseline (0.319)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_data['Model'], fontsize=9)
axes[0].set_ylabel('Test Set Accuracy')
axes[0].set_title('Multi-class Model Comparison: Test Accuracy')
for i, (acc, status) in enumerate(zip(comparison_data['Test Acc'],
                                       comparison_data['Status'])):
    axes[0].text(i, acc + 0.015, f'{acc:.3f}\n({status})',
                 ha='center', fontsize=8.5)
axes[0].legend(loc='upper left'); axes[0].set_ylim(0, 0.6)
axes[0].grid(alpha=0.3, axis='y')

# Macro F1 comparison
axes[1].bar(x, comparison_data['Macro F1'], color=colors, edgecolor='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels(comparison_data['Model'], fontsize=9)
axes[1].set_ylabel('Macro-Averaged F1 Score')
axes[1].set_title('Multi-class Model Comparison: Macro F1')
for i, f1 in enumerate(comparison_data['Macro F1']):
    axes[1].text(i, f1 + 0.01, f'{f1:.3f}', ha='center', fontsize=10)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIG_DIR / '13_model_comparison.png')
plt.show()
print("Saved model comparison plot.")


