"""
PFSC: Preprocessing, First-order and Second-order Classification Framework
for Electricity Theft Detection in Smart Grids

Based on:
"A Stacked Machine and Deep Learning-Based Approach for Analysing Electricity Theft in Smart Grids"
IEEE Transactions on Smart Grid, Vol. 13, No. 2, March 2022
Authors: Inam Ullah Khan, Nadeem Javeid, C. James Taylor, Kelum A.A. Gamage, Xiandong Ma

Dataset: SGCC (State Grid Corporation of China)
Source: https://github.com/henryRDlab/ElectricityTheftDetection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, ConfusionMatrixDisplay)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping


# =============================================================================
# MODULE 1: DATA PREPARATION (IONB)
# =============================================================================

def load_sgcc_data(filepath):
    """
    Load SGCC dataset.
    Expected format: rows = users, columns = daily consumption + label (last col).
    Download from: https://github.com/henryRDlab/ElectricityTheftDetection
    """
    df = pd.read_csv(filepath, index_col=0)
    labels = df.iloc[:, -1].values
    features = df.iloc[:, :-1].values
    print(f"Dataset shape: {features.shape}, Labels: {np.bincount(labels.astype(int))}")
    return features, labels


def impute_missing_values(X):
    """
    Module 1a: Recover missing values using linear interpolation.
    - Remove users with >600 null values
    - Interpolate if missing < 7 samples
    - Replace with 0 if missing >= 7
    """
    df = pd.DataFrame(X)
    null_counts = df.isnull().sum(axis=1)
    df = df[null_counts <= 600].reset_index(drop=True)

    for i in range(len(df)):
        n_missing = df.iloc[i].isnull().sum()
        if n_missing == 0:
            continue
        elif n_missing < 7:
            df.iloc[i] = df.iloc[i].interpolate(method='linear', limit_direction='both')
        else:
            df.iloc[i] = df.iloc[i].fillna(0)

    print(f"After imputation: {df.shape[0]} users retained")
    return df.values


def handle_outliers(X):
    """
    Module 1b: Three-sigma rule to handle outliers.
    Values beyond mean + 2*std are clipped.
    """
    X_clean = X.copy().astype(float)
    for i in range(X_clean.shape[0]):
        row = X_clean[i]
        mean_val = np.nanmean(row)
        std_val = np.nanstd(row)
        threshold = mean_val + 2 * std_val
        X_clean[i] = np.where(row > threshold, threshold, row)
    return X_clean


def normalize_data(X):
    """
    Module 1c: Min-Max normalization to scale values in [0, 1].
    """
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X.T).T  # normalize per feature (column)
    return X_norm, scaler


def apply_stlu_resampling(X, y):
    """
    Module 1d: STLU - SMOTE + Tomek Links Undersampling
    Combines oversampling (SMOTE) and undersampling (Tomek Links)
    to handle class imbalance in theft detection.
    """
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('tomek', TomekLinks())
    ])
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    print(f"After STLU resampling - Class distribution: {np.bincount(y_resampled.astype(int))}")
    return X_resampled, y_resampled


def ionb_pipeline(X, y):
    """
    Full IONB: Imputation → Outlier Handling → Normalization → Class Balancing
    """
    print("\n=== MODULE 1: IONB DATA PREPARATION ===")

    # Filter labels to match imputed data rows
    df = pd.DataFrame(X)
    null_counts = df.isnull().sum(axis=1)
    valid_mask = null_counts <= 600
    y_filtered = y[valid_mask.values]

    X1 = impute_missing_values(X)
    X2 = handle_outliers(X1)
    X3, scaler = normalize_data(X2)
    X4, y4 = apply_stlu_resampling(X3, y_filtered)

    print(f"IONB complete. Final shape: {X4.shape}")
    return X4, y4, scaler


# =============================================================================
# MODULE 2: FIRST-ORDER BASE CLASSIFIERS
# =============================================================================

def train_base_classifiers(X_train, y_train):
    """
    Module 2: Train three uncorrelated base classifiers in parallel.
    - SVM (Support Vector Machine)
    - RF  (Random Forest)
    - GBDT (Gradient Boosting Decision Tree)
    """
    print("\n=== MODULE 2: FIRST-ORDER BASE CLASSIFIERS ===")

    classifiers = {
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        'RF':  RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'GBDT': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    trained = {}
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        clf.fit(X_train, y_train)
        trained[name] = clf
        print(f"  {name} trained ✓")

    return trained


def get_base_predictions(classifiers, X):
    """
    Get probability predictions from each base classifier.
    These form the new feature set for the meta-classifier.
    """
    preds = []
    for name, clf in classifiers.items():
        prob = clf.predict_proba(X)[:, 1]
        preds.append(prob)
    return np.column_stack(preds)  # shape: (n_samples, 3)


# =============================================================================
# MODULE 3: SECOND-ORDER META-CLASSIFIER (TCN)
# =============================================================================

def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    """
    TCN Residual Block with:
    - Dilated causal convolution
    - Batch normalization
    - Dropout regularization
    - Residual connection
    """
    # First causal dilated conv layer
    conv1 = layers.Conv1D(filters, kernel_size, padding='causal',
                          dilation_rate=dilation_rate, activation='relu')(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(dropout_rate)(conv1)

    # Second causal dilated conv layer
    conv2 = layers.Conv1D(filters, kernel_size, padding='causal',
                          dilation_rate=dilation_rate, activation='relu')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(dropout_rate)(conv2)

    # Residual connection (adjust dimensions if needed)
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding='same')(x)

    return layers.Add()([x, conv2])


def build_tcn_model(input_dim=3, num_filters=64, kernel_size=2,
                    num_blocks=3, dropout_rate=0.2):
    """
    Build Temporal Convolutional Network (TCN) for meta-classification.

    Architecture:
    - Input: 3 features (predictions from 3 base classifiers)
    - 3 TCN blocks with exponentially increasing dilation
    - Dense output layer with sigmoid activation
    """
    inputs = layers.Input(shape=(input_dim, 1))
    x = inputs

    # Stack TCN blocks with exponentially growing dilation: 1, 2, 4
    for i in range(num_blocks):
        dilation = 2 ** i
        x = residual_block(x, num_filters, kernel_size, dilation, dropout_rate)

    # Global pooling and classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='TCN_MetaClassifier')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_tcn(X_meta_train, y_train, X_meta_val, y_val, epochs=20, batch_size=64):
    """
    Train TCN meta-classifier on base classifier outputs.
    Input shape: (n_samples, 3, 1) — 3 base classifier predictions
    """
    print("\n=== MODULE 3: TCN META-CLASSIFIER ===")

    # Reshape for Conv1D: (samples, timesteps=3, channels=1)
    X_train_tcn = X_meta_train.reshape(-1, 3, 1)
    X_val_tcn = X_meta_val.reshape(-1, 3, 1)

    model = build_tcn_model()
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_tcn, y_train,
        validation_data=(X_val_tcn, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history


# =============================================================================
# EVALUATION AND VISUALIZATION
# =============================================================================

def evaluate_model(name, y_true, y_pred_prob, threshold=0.5):
    """Compute and print all evaluation metrics."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    metrics = {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0),
        'AUC':       roc_auc_score(y_true, y_pred_prob)
    }
    print(f"\n--- {name} Performance ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def plot_results(history, base_classifiers, X_meta_test, y_test, tcn_model):
    """Generate all plots from the paper."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PFSC Electricity Theft Detection Results', fontsize=14, fontweight='bold')

    # 1. Training history (accuracy & loss)
    ax1 = axes[0, 0]
    ax1.plot(history.history['accuracy'], 'b-o', label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], 'r-o', label='Val Accuracy')
    ax1.set_title('TCN Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[0, 1]
    ax2.plot(history.history['loss'], 'b-o', label='Train Loss')
    ax2.plot(history.history['val_loss'], 'r-o', label='Val Loss')
    ax2.set_title('TCN Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # 2. ROC Curves for all classifiers
    ax3 = axes[1, 0]
    colors = {'SVM': 'green', 'RF': 'blue', 'GBDT': 'orange', 'TCN (PFSC)': 'red'}

    for name, clf in base_classifiers.items():
        prob = clf.predict_proba(X_meta_test)[:, 1] if hasattr(clf, 'predict_proba') else None
        # Use base predictions stored in X_meta_test
    
    # TCN ROC
    X_tcn = X_meta_test.reshape(-1, 3, 1)
    tcn_prob = tcn_model.predict(X_tcn, verbose=0).flatten()
    fpr, tpr, _ = roc_curve(y_test, tcn_prob)
    auc_score = roc_auc_score(y_test, tcn_prob)
    ax3.plot(fpr, tpr, color='red', lw=2, label=f'TCN PFSC (AUC={auc_score:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', label='Random')
    ax3.set_title('ROC Curve - PFSC (TCN)')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend()
    ax3.grid(True)

    # 3. Confusion Matrix
    ax4 = axes[1, 1]
    y_pred = (tcn_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Honest', 'Theft'])
    disp.plot(ax=ax4, colorbar=False, cmap='Blues')
    ax4.set_title('Confusion Matrix (TCN Meta-Classifier)')

    plt.tight_layout()
    plt.savefig('pfsc_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResults saved to pfsc_results.png")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pfsc_pipeline(X, y, test_size=0.2):
    """
    Full PFSC pipeline:
    1. IONB Data Preparation
    2. First-order Base Classifiers (SVM, RF, GBDT)
    3. Second-order Meta-Classifier (TCN)
    4. Evaluation
    """
    print("=" * 60)
    print("  PFSC ELECTRICITY THEFT DETECTION FRAMEWORK")
    print("  IEEE Transactions on Smart Grid, 2022")
    print("=" * 60)

    # Step 1: IONB preprocessing
    X_prepared, y_prepared, scaler = ionb_pipeline(X, y)

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_prepared, y_prepared, test_size=test_size, random_state=42, stratify=y_prepared
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # Step 3: Train base classifiers
    base_classifiers = train_base_classifiers(X_train, y_train)

    # Step 4: Generate meta-features (base classifier outputs)
    X_meta_train = get_base_predictions(base_classifiers, X_train)
    X_meta_val   = get_base_predictions(base_classifiers, X_val)
    X_meta_test  = get_base_predictions(base_classifiers, X_test)

    # Step 5: Evaluate base classifiers
    print("\n=== BASE CLASSIFIER EVALUATION ON TEST SET ===")
    for name, clf in base_classifiers.items():
        prob = clf.predict_proba(X_test)[:, 1]
        evaluate_model(name, y_test, prob)

    # Step 6: Train TCN meta-classifier
    tcn_model, history = train_tcn(X_meta_train, y_train, X_meta_val, y_val)

    # Step 7: Evaluate PFSC (full framework)
    print("\n=== PFSC FULL FRAMEWORK EVALUATION ===")
    X_tcn_test = X_meta_test.reshape(-1, 3, 1)
    tcn_prob = tcn_model.predict(X_tcn_test, verbose=0).flatten()
    evaluate_model("PFSC (TCN Meta-Classifier)", y_test, tcn_prob)

    # Step 8: Plot results
    plot_results(history, base_classifiers, X_meta_test, y_test, tcn_model)

    return tcn_model, base_classifiers, scaler


# =============================================================================
# DEMO WITH SYNTHETIC DATA (if SGCC not available)
# =============================================================================

def generate_synthetic_sgcc(n_honest=3876, n_theft=390, n_features=1035):
    """
    Generate synthetic data resembling SGCC distribution (90% honest, 10% theft)
    for demonstration when real SGCC data is not available.
    """
    print("\n[INFO] Using synthetic data. Replace with real SGCC data for paper results.")
    np.random.seed(42)

    # Honest consumers: smooth, regular patterns
    X_honest = np.random.normal(loc=5.0, scale=1.5, size=(n_honest, n_features))
    X_honest = np.clip(X_honest, 0, None)

    # Theft consumers: irregular patterns with anomalies
    X_theft = np.random.normal(loc=3.0, scale=3.0, size=(n_theft, n_features))
    X_theft = np.clip(X_theft, 0, None)

    # Add some NaN values to simulate real missing data
    mask = np.random.random(X_honest.shape) < 0.01
    X_honest[mask] = np.nan

    X = np.vstack([X_honest, X_theft])
    y = np.array([0] * n_honest + [1] * n_theft)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


if __name__ == "__main__":
    # -------------------------------------------------------
    # Option A: Use real SGCC data
    # Download from: https://github.com/henryRDlab/ElectricityTheftDetection
    # Uncomment below and set correct path:
    # X, y = load_sgcc_data("data/sgcc_data.csv")
    # -------------------------------------------------------

    # Option B: Run with synthetic data for demonstration
    X, y = generate_synthetic_sgcc(n_honest=2000, n_theft=200, n_features=100)

    # Run full PFSC pipeline
    tcn_model, base_classifiers, scaler = run_pfsc_pipeline(X, y)

    # Save TCN model
    tcn_model.save("pfsc_tcn_model.h5")
    print("\nModel saved to pfsc_tcn_model.h5")
    print("\nDone! ✓")
