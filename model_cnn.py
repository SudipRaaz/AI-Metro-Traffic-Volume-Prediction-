# =============================================================================
# PRO502 - Assignment 2
# File: model_cnn.py
# Step 2: Model - Convolutional Neural Network (CNN) for Time-Series
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import preprocess


# --- Robust coercion & name-resolution for CNN training inputs ---
import numpy as np

def ensure_numeric_array(x):
    """Return numpy float32 array from DataFrame/Series/array-like, robustly handling object dtype."""
    if x is None:
        return None
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float32)
        except Exception:
            # elementwise fallback
            arr = np.array([float(v) if v not in (None, "", "nan", "NaN") else np.nan
                            for v in arr.flat], dtype=np.float32).reshape(arr.shape)
    else:
        arr = arr.astype(np.float32)
    return arr

# Possible variable name tuples to search (train, val, test)
candidate_names = [
    ("X_train_windows", "y_train_windows", "X_val_windows", "y_val_windows", "X_test_windows", "y_test_windows"),
    ("X_windows",       "y_windows",       "X_val_windows", "y_val_windows", "X_test_windows", "y_test_windows"),
    ("X_train_seq",     "y_train_seq",     "X_val_seq",     "y_val_seq",     "X_test_seq",     "y_test_seq"),
    ("X_train",         "y_train",         "X_val",         "y_val",         "X_test",         "y_test"),
]

found = None
g = globals()
for names in candidate_names:
    if names[0] in g and names[1] in g:
        found = names
        break

if found is None:
    # Last attempt: maybe arrays are local variables; try to refer to common names and raise helpful error
    raise NameError("Could not find expected train/val/test variables (e.g. X_train_windows, X_windows, X_train_seq). "
                    "Open model_cnn.py and check the variable names used for the sliding windows and adjust the candidate_names list accordingly.")

# Extract and coerce
X_train_raw = g.get(found[0])
y_train_raw = g.get(found[1])
X_val_raw   = g.get(found[2])
y_val_raw   = g.get(found[3])
X_test_raw  = g.get(found[4])
y_test_raw  = g.get(found[5])

X_train = ensure_numeric_array(X_train_raw)
y_train = ensure_numeric_array(y_train_raw).reshape(-1)
X_val   = ensure_numeric_array(X_val_raw)
y_val   = ensure_numeric_array(y_val_raw).reshape(-1)
X_test  = ensure_numeric_array(X_test_raw)
y_test  = ensure_numeric_array(y_test_raw).reshape(-1)

# Final sanity prints / asserts
print("After coercion:")
print("  X_train:", None if X_train is None else (X_train.shape, X_train.dtype))
print("  y_train:", None if y_train is None else (y_train.shape, y_train.dtype))
print("  X_val  :", None if X_val is None   else (X_val.shape, X_val.dtype))
print("  y_val  :", None if y_val is None   else (y_val.shape, y_val.dtype))

if X_train is None or y_train is None:
    raise RuntimeError("Coercion failed: X_train or y_train is None. Check variable names for windows in model_cnn.py")

assert X_train.dtype == np.float32, f"X_train dtype is {X_train.dtype}"
assert y_train.dtype == np.float32, f"y_train dtype is {y_train.dtype}"
assert X_train.shape[0] == y_train.shape[0], f"Row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}"

# =============================================================================
# STEP A: CREATE SLIDING WINDOWS
# =============================================================================

def create_windows(X, y, window_size=24):
    """
    CNNs don't take single rows — they take SEQUENCES (windows) of rows.

    Why? Because traffic at hour 8 depends on what happened at hours 1-7.
    A window gives the model that temporal context.

    Example with window_size=24:
        Input:  24 consecutive hours of all features
        Output: traffic volume at the NEXT hour

    Sliding window process:
        Window 1: rows[0:24]   → predict y[24]
        Window 2: rows[1:25]   → predict y[25]
        Window 3: rows[2:26]   → predict y[26]
        ...

    Result shapes:
        X_windows: (num_samples, window_size, num_features)
        y_windows: (num_samples,)

    Think of X_windows like a stack of "screenshots" —
    each screenshot shows 24 hours of all features.
    The CNN learns to read each screenshot and predict
    what comes next.
    """
    X_windows = []
    y_windows = []

    X_values = X.values  # Convert DataFrame to numpy array
    y_values = y.values

    for i in range(len(X_values) - window_size):
        # Grab window_size consecutive rows
        window = X_values[i : i + window_size]
        # Target is the traffic value RIGHT AFTER the window
        target = y_values[i + window_size]

        X_windows.append(window)
        y_windows.append(target)

    X_windows = np.array(X_windows)  # Shape: (samples, 24, num_features)
    y_windows = np.array(y_windows)  # Shape: (samples,)

    print(f"Window size      : {window_size} hours")
    print(f"X_windows shape  : {X_windows.shape}  (samples, time_steps, features)")
    print(f"y_windows shape  : {y_windows.shape}")

    return X_windows, y_windows


# =============================================================================
# STEP B: BUILD CNN ARCHITECTURE
# =============================================================================

def build_cnn(window_size, num_features, filters=64, kernel_size=3):
    """
    Build the CNN architecture layer by layer.

    INPUT SHAPE: (window_size, num_features)
    e.g. (24, 24) = 24 time steps, 24 features each

    LAYER BY LAYER EXPLANATION:

    Conv1D(filters=64, kernel_size=3):
        - Slides a filter of width 3 (3 time steps) across the window
        - Like reading 3 hours at a time and asking "what pattern is here?"
        - 64 filters = looks for 64 different patterns simultaneously
        - Output: (22, 64) — 22 positions × 64 patterns found

    ReLU activation:
        - Applied automatically via activation='relu'
        - Formula: output = max(0, input)
        - Turns negative values to 0 (model ignores irrelevant patterns)
        - Adds non-linearity so model can learn complex relationships

    MaxPooling1D(pool_size=2):
        - Takes the MAXIMUM value from every pair of positions
        - Compresses (22, 64) → (11, 64)
        - Keeps the strongest signal, discards the weak ones
        - Also helps with translation invariance
          (rush hour at 8am or 9am = still rush hour)

    Second Conv1D(filters=32):
        - Looks for higher-level patterns in the compressed output
        - Like combining "morning low + sharp spike = rush hour"

    Flatten():
        - Conv1D output is 2D (time_steps, filters)
        - Dense layers need 1D input
        - Flatten just unrolls 2D → 1D

    Dense(64):
        - Fully connected layer
        - Combines ALL detected patterns into a prediction

    Dropout(0.2):
        - Randomly turns off 20% of neurons during TRAINING only
        - Forces the model to not rely on any single neuron
        - Prevents overfitting — very important for neural networks

    Dense(1):
        - Final output layer — predicts ONE value (traffic_volume)
        - No activation = raw number output (regression)
    """
    model = Sequential([
        # First convolutional block
        Conv1D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               input_shape=(window_size, num_features),
               padding='same'),   # 'same' keeps output same length as input
        MaxPooling1D(pool_size=2),

        # Second convolutional block (detects higher-level patterns)
        Conv1D(filters=filters // 2,
               kernel_size=kernel_size,
               activation='relu',
               padding='same'),
        MaxPooling1D(pool_size=2),

        # Convert 2D feature maps to 1D vector
        Flatten(),

        # Fully connected layers
        Dense(64, activation='relu'),
        Dropout(0.2),   # Drop 20% of neurons randomly during training

        # Output layer — single predicted traffic value
        Dense(1)
    ])

    # Compile: define loss function and optimiser
    # mean_squared_error loss = penalises large prediction errors more
    # adam optimiser = adaptive learning rate, works well for most problems
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Print a summary of the architecture
    model.summary()

    return model


# =============================================================================
# STEP C: TRAIN THE CNN
# =============================================================================

def train_cnn(model, X_train_w, y_train_w, X_val_w, y_val_w,
              epochs=50, batch_size=64):
    """
    Train the CNN on windowed data.

    epochs=50:
        The model sees the FULL training set 50 times.
        Each pass = one epoch.
        More epochs = more learning, but risk of overfitting.

    batch_size=64:
        Instead of updating weights after EVERY window (slow)
        or after ALL windows (memory intensive),
        we update after every 64 windows (good balance).

    EarlyStopping:
        Monitors val_loss every epoch.
        If val_loss hasn't improved for 'patience' epochs → stop training.
        Restores the weights from the best epoch automatically.
        This prevents overfitting AND saves training time.
        This is automatic hyperparameter tuning for epochs!
    """
    print("\n--- Training CNN ---")

    early_stop = EarlyStopping(
        monitor='val_loss',   # Watch validation loss
        patience=5,           # Stop if no improvement for 5 epochs
        restore_best_weights=True,  # Revert to best epoch's weights
        verbose=1
    )

    # if earlier code used X_windows/y_windows variables, rebinding is fine:
    X_windows = X_train_windows
    y_windows = y_train_windows

    history = model.fit(
        X_windows, y_windows,
        validation_data=(X_val_windows, y_val_windows),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1   # Print progress each epoch
    )

    print("\nTraining complete.")
    return model, history


# =============================================================================
# STEP D: EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X_w, y_w, split_name="Validation"):
    """
    Same metrics as all previous models for fair comparison.
    MAE, RMSE, R²
    """
    predictions = model.predict(X_w).flatten()  # Flatten (n,1) → (n,)

    mae  = mean_absolute_error(y_w, predictions)
    rmse = np.sqrt(mean_squared_error(y_w, predictions))
    r2   = r2_score(y_w, predictions)

    print(f"\n--- {split_name} Results ---")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "predictions": predictions}


# =============================================================================
# STEP E: PLOT TRAINING HISTORY
# =============================================================================

def plot_training_history(history):
    """
    Shows train loss vs val loss over each epoch.

    What to look for:
    - Both lines decreasing = model is learning correctly
    - Val loss much higher than train loss = overfitting
    - Val loss stops improving (flat) = early stopping kicked in correctly
    - Val loss starts RISING while train still falls = overfitting started here

    This plot is essential for your report — it shows the
    training process visually and justifies early stopping.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(history.history['loss'],     label='Train Loss', color='steelblue')
    plt.plot(history.history['val_loss'], label='Val Loss',   color='tomato')

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("CNN Training History: Loss per Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cnn_training_history.png", dpi=150)
    plt.show()
    print("Plot saved as: cnn_training_history.png")


# =============================================================================
# STEP F: PLOT PREDICTED VS ACTUAL
# =============================================================================

def plot_predictions(y_actual, y_predicted, n_points=500):
    """
    Two plots:
    1. Scatter: predicted vs actual (tighter = better)
    2. Line: predicted vs actual over time (shows if CNN tracks the pattern)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Scatter
    axes[0].scatter(y_actual[:n_points], y_predicted[:n_points],
                    alpha=0.3, color='darkorange', s=5)
    axes[0].plot([y_actual.min(), y_actual.max()],
                 [y_actual.min(), y_actual.max()],
                 'r--', linewidth=2, label='Perfect line')
    axes[0].set_xlabel("Actual Traffic Volume")
    axes[0].set_ylabel("Predicted Traffic Volume")
    axes[0].set_title("CNN: Predicted vs Actual (scatter)")
    axes[0].legend()

    # Plot 2: Time series line
    axes[1].plot(y_actual[:n_points],    label='Actual',    alpha=0.7,
                 color='steelblue', linewidth=1)
    axes[1].plot(y_predicted[:n_points], label='Predicted', alpha=0.7,
                 color='tomato',    linewidth=1)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Traffic Volume")
    axes[1].set_title("CNN: Predicted vs Actual (over time)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cnn_predictions.png", dpi=150)
    plt.show()
    print("Plot saved as: cnn_predictions.png")


# =============================================================================
# RUN THIS FILE DIRECTLY
# =============================================================================

if __name__ == "__main__":

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # -------------------------------------------------------------------------
    # 1. Load preprocessed data
    # -------------------------------------------------------------------------
    print("=" * 55)
    print("Loading preprocessed data...")
    print("=" * 55)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess()

    # -------------------------------------------------------------------------
    # 2. Create sliding windows
    # -------------------------------------------------------------------------
    print("\n--- Creating Sliding Windows ---")
    WINDOW_SIZE = 24  # 24 hours = one full day of context

    X_train_w, y_train_w = create_windows(X_train, y_train, WINDOW_SIZE)
    X_val_w,   y_val_w   = create_windows(X_val,   y_val,   WINDOW_SIZE)
    X_test_w,  y_test_w  = create_windows(X_test,  y_test,  WINDOW_SIZE)

    num_features = X_train_w.shape[2]
    print(f"\nNum features per time step: {num_features}")

    # -------------------------------------------------------------------------
    # 3. Build CNN architecture
    # -------------------------------------------------------------------------
    print("\n--- Building CNN Architecture ---")
    model = build_cnn(
        window_size  = WINDOW_SIZE,
        num_features = num_features,
        filters      = 64,
        kernel_size  = 3
    )

    # -------------------------------------------------------------------------
    # 4. Train CNN
    # -------------------------------------------------------------------------
    model, history = train_cnn(
        model,
        X_train_w, y_train_w,
        X_val_w,   y_val_w,
        epochs     = 50,
        batch_size = 64
    )

    # -------------------------------------------------------------------------
    # 5. Evaluate on validation set
    # -------------------------------------------------------------------------
    val_results = evaluate_model(model, X_val_w, y_val_w, "Validation")

    # -------------------------------------------------------------------------
    # 6. Plots
    # -------------------------------------------------------------------------
    plot_training_history(history)
    plot_predictions(y_val_w, val_results["predictions"])

    # -------------------------------------------------------------------------
    # 7. Comparison summary
    # -------------------------------------------------------------------------
    print("\n--- Model Comparison So Far ---")
    print(f"Linear Regression  Val R²: ~0.40-0.55  (baseline)")
    print(f"Decision Tree      Val R²: ~0.80-0.85  (overfits)")




    print("Next: LSTM — designed specifically for sequences like this.")
    print("\nCNN should compete with Random Forest.")
    print(f"CNN                Val R²: {val_results['R2']:.4f}         (deep learning)")
    print(f"Random Forest      Val R²: ~0.87-0.91  (best traditional)")