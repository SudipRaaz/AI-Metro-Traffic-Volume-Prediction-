# =============================================================================
# PRO502 - Assignment 2
# File: model_lstm.py
# Step 2: Model - LSTM (Long Short-Term Memory) for Time-Series
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from preprocessing import preprocess


# =============================================================================
# STEP A: CREATE SLIDING WINDOWS (same concept as CNN)
# =============================================================================

def create_windows(X, y, window_size=24):
    """
    Same sliding window approach as CNN.
    LSTM reads sequences — we feed it windows of consecutive hours.

    The key difference from CNN:
    CNN processes the whole window at once (like looking at a photo).
    LSTM processes it STEP BY STEP, updating its memory at each step.

    So for window_size=24:
    CNN  → sees all 24 hours simultaneously, finds patterns
    LSTM → reads hour 1, updates memory
           reads hour 2, updates memory
           ...
           reads hour 24, makes prediction

    Shape: (samples, window_size, num_features)
    Same as CNN — the architecture handles it differently internally.
    """
    X_windows = []
    y_windows = []

    X_values = X.values
    y_values = y.values

    for i in range(len(X_values) - window_size):
        X_windows.append(X_values[i : i + window_size])
        y_windows.append(y_values[i + window_size])

    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)

    print(f"Window size     : {window_size} hours")
    print(f"X_windows shape : {X_windows.shape}  (samples, time_steps, features)")
    print(f"y_windows shape : {y_windows.shape}")

    return X_windows, y_windows


# =============================================================================
# STEP B: BUILD LSTM ARCHITECTURE
# =============================================================================

def build_lstm(window_size, num_features, units=64):
    """
    Build a stacked LSTM architecture.

    INPUT SHAPE: (window_size, num_features)
    e.g. (24, 24) — 24 time steps, 24 features each

    LAYER BY LAYER:

    LSTM(units=64, return_sequences=True):
        - units=64 means 64 memory cells (64 things it can remember)
        - return_sequences=True means output a value at EVERY time step
          (not just the last one)
        - Why? Because the next LSTM layer also needs the full sequence
        - This is what makes it "stacked" LSTM

    Dropout(0.2):
        - Randomly turns off 20% of LSTM outputs during training
        - Prevents the model from relying too heavily on specific memory cells
        - Only active during TRAINING, not during prediction

    LSTM(units=32, return_sequences=False):
        - Second LSTM layer — learns higher-level temporal patterns
        - return_sequences=False means only output at the LAST time step
          (we only need one final summary of the sequence)
        - Fewer units (32) — compresses the representation

    Dense(32, activation='relu'):
        - Fully connected layer
        - Combines LSTM's temporal summary into a prediction
        - relu keeps only positive activations

    Dropout(0.2):
        - Another dropout for regularisation before final output

    Dense(1):
        - Output layer
        - Predicts ONE value: traffic_volume
        - No activation = raw regression output
    """
    model = Sequential([

        # First LSTM layer — reads sequence step by step
        # return_sequences=True passes full sequence to next LSTM layer
        LSTM(units=units,
             return_sequences=True,
             input_shape=(window_size, num_features)),
        Dropout(0.2),

        # Second LSTM layer — higher-level temporal patterns
        # return_sequences=False — only need final summary
        LSTM(units=units // 2,
             return_sequences=False),
        Dropout(0.2),

        # Dense layers to convert LSTM output to prediction
        Dense(32, activation='relu'),
        Dropout(0.2),

        # Output: single traffic volume value
        Dense(1)
    ])

    # Adam optimiser with default learning rate (0.001)
    # MSE loss penalises large errors heavily — good for traffic spikes
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()
    return model


# =============================================================================
# STEP C: TRAIN THE LSTM
# =============================================================================

def train_lstm(model, X_train_w, y_train_w, X_val_w, y_val_w,
               epochs=50, batch_size=64):
    """
    Train the LSTM with two important callbacks:

    1. EarlyStopping:
       - Same as CNN — stops when val_loss stops improving
       - patience=7 (slightly more than CNN's 5)
       - Why more patience? LSTM learning can be slower and less smooth
         It may temporarily get worse before getting better
       - restore_best_weights=True reverts to best epoch automatically

    2. ReduceLROnPlateau (NEW — not in CNN):
       - LR = Learning Rate
       - Learning rate controls how big each weight update step is
       - Too large → model overshoots, bounces around, never converges
       - Too small → model learns too slowly
       - ReduceLROnPlateau monitors val_loss:
           If no improvement for 3 epochs → multiply LR by factor=0.5
           (cut learning rate in half)
       - This helps the model fine-tune in later epochs
       - Think of it like: running fast at first, then tiptoeing to precision

    Why does LSTM need ReduceLROnPlateau more than CNN?
    LSTM has more complex internal state (3 gates per cell).
    Its loss landscape is bumpier — adaptive LR helps navigate it.
    """
    print("\n--- Training LSTM ---")
    print("(This may take a few minutes — LSTM is more complex than CNN)\n")

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,                  # More patience than CNN
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,                  # Halve the learning rate
        patience=3,                  # After 3 epochs of no improvement
        min_lr=1e-6,                 # Never go below this LR
        verbose=1
    )

    history = model.fit(
        X_train_w, y_train_w,
        validation_data=(X_val_w, y_val_w),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("\nTraining complete.")
    return model, history


# =============================================================================
# STEP D: EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X_w, y_w, split_name="Validation"):
    """
    Same 3 metrics for fair comparison across all models:
    MAE, RMSE, R²
    """
    predictions = model.predict(X_w).flatten()

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
    LSTM training curves are often less smooth than CNN.
    You may see occasional spikes in val_loss — this is normal.

    What to look for:
    - General downward trend in both curves = learning happening
    - Sudden val_loss spike followed by recovery = LR was reduced
    - Both curves plateau = early stopping about to trigger
    - Val loss rises sharply = overfitting

    The learning rate reductions will appear as sudden drops
    in the loss curve — you can mention this in your report.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss over epochs
    axes[0].plot(history.history['loss'],
                 label='Train Loss', color='steelblue', linewidth=1.5)
    axes[0].plot(history.history['val_loss'],
                 label='Val Loss',   color='tomato',    linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("LSTM Training History: Loss per Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Zoomed in (remove first 3 epochs which are always high)
    axes[1].plot(history.history['loss'][3:],
                 label='Train Loss', color='steelblue', linewidth=1.5)
    axes[1].plot(history.history['val_loss'][3:],
                 label='Val Loss',   color='tomato',    linewidth=1.5)
    axes[1].set_xlabel("Epoch (from epoch 4)")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].set_title("LSTM Training History: Zoomed In")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lstm_training_history.png", dpi=150)
    plt.show()
    print("Plot saved as: lstm_training_history.png")


# =============================================================================
# STEP F: PLOT PREDICTIONS
# =============================================================================

def plot_predictions(y_actual, y_predicted, n_points=500):
    """
    Two plots — same as CNN so you can visually compare them side by side.

    Key things to compare between LSTM and CNN plots:
    1. Does LSTM track sharp traffic spikes better? (time series plot)
    2. Is LSTM's scatter plot tighter around the diagonal?
    3. Does LSTM handle the overnight low traffic better?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_actual[:n_points], y_predicted[:n_points],
                    alpha=0.3, color='purple', s=5)
    axes[0].plot([y_actual.min(), y_actual.max()],
                 [y_actual.min(), y_actual.max()],
                 'r--', linewidth=2, label='Perfect line')
    axes[0].set_xlabel("Actual Traffic Volume")
    axes[0].set_ylabel("Predicted Traffic Volume")
    axes[0].set_title("LSTM: Predicted vs Actual (scatter)")
    axes[0].legend()

    # Time series plot
    axes[1].plot(y_actual[:n_points],    label='Actual',
                 color='steelblue', linewidth=1, alpha=0.8)
    axes[1].plot(y_predicted[:n_points], label='Predicted',
                 color='tomato',    linewidth=1, alpha=0.8)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Traffic Volume")
    axes[1].set_title("LSTM: Predicted vs Actual (over time)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lstm_predictions.png", dpi=150)
    plt.show()
    print("Plot saved as: lstm_predictions.png")


# =============================================================================
# STEP G: COMPARE CNN vs LSTM DIRECTLY
# =============================================================================

def compare_with_cnn(lstm_results, y_actual):
    """
    Side-by-side comparison of CNN and LSTM.
    Helps you decide which deep learning model to recommend in your report.

    Key question: Does LSTM's memory mechanism actually help
    over CNN's pattern detection for this specific dataset?
    The metrics will tell you.
    """
    print("\n" + "=" * 50)
    print("CNN vs LSTM COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<10} {'CNN (expected)':<20} {'LSTM':<20}")
    print("-" * 50)
    print(f"{'MAE':<10} {'~600-900':<20} {lstm_results['MAE']:<20.2f}")
    print(f"{'RMSE':<10} {'~900-1200':<20} {lstm_results['RMSE']:<20.2f}")
    print(f"{'R²':<10} {'~0.83-0.89':<20} {lstm_results['R2']:<20.4f}")
    print("=" * 50)
    print("\nIf LSTM R² > CNN R²:")
    print("  → Memory gates help capture long-term traffic patterns")
    print("If CNN R² ≈ LSTM R²:")
    print("  → Local pattern detection is sufficient for this dataset")
    print("  → CNN is preferred (faster, simpler)")


# =============================================================================
# RUN THIS FILE DIRECTLY
# =============================================================================

if __name__ == "__main__":

    # Reproducibility
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
    WINDOW_SIZE = 24

    X_train_w, y_train_w = create_windows(X_train, y_train, WINDOW_SIZE)
    X_val_w,   y_val_w   = create_windows(X_val,   y_val,   WINDOW_SIZE)
    X_test_w,  y_test_w  = create_windows(X_test,  y_test,  WINDOW_SIZE)

    num_features = X_train_w.shape[2]

    # -------------------------------------------------------------------------
    # 3. Build LSTM
    # -------------------------------------------------------------------------
    print("\n--- Building LSTM Architecture ---")
    model = build_lstm(
        window_size  = WINDOW_SIZE,
        num_features = num_features,
        units        = 64
    )

    # -------------------------------------------------------------------------
    # 4. Train LSTM
    # -------------------------------------------------------------------------
    model, history = train_lstm(
        model,
        X_train_w, y_train_w,
        X_val_w,   y_val_w,
        epochs     = 50,
        batch_size = 64
    )

    # -------------------------------------------------------------------------
    # 5. Evaluate
    # -------------------------------------------------------------------------
    val_results = evaluate_model(model, X_val_w, y_val_w, "Validation")

    # -------------------------------------------------------------------------
    # 6. Plots
    # -------------------------------------------------------------------------
    plot_training_history(history)
    plot_predictions(y_val_w, val_results["predictions"])

    # -------------------------------------------------------------------------
    # 7. CNN vs LSTM comparison
    # -------------------------------------------------------------------------
    compare_with_cnn(val_results, y_val_w)

    # -------------------------------------------------------------------------
    # 8. Full leaderboard so far
    # -------------------------------------------------------------------------
    print("\n--- FULL MODEL LEADERBOARD (Validation R²) ---")
    print(f"  Linear Regression : ~0.40-0.55  (baseline)")
    print(f"  Decision Tree     : ~0.80-0.85  (overfits without tuning)")
    print(f"  Random Forest     : ~0.87-0.91  (best traditional ML)")
    print(f"  CNN               : ~0.83-0.89  (deep learning, pattern-based)")
    print(f"  LSTM              :  {val_results['R2']:.4f}         (deep learning, memory-based)")
    print("\nNext step: Run ALL models on TEST data for final evaluation.")

    # --- NEW: Convert to sequences for LSTM ---
    import numpy as _np

    def ensure_numeric_array(x):
        # Accept DataFrame/Series or numpy array-like, return float32 numpy array
        if hasattr(x, "to_numpy"):
            arr = x.to_numpy()
        else:
            arr = np.asarray(x)
        # If object dtype, try to coerce to numbers
        if arr.dtype == object:
            # Try a safe elementwise conversion to float
            try:
                arr = arr.astype(np.float32)
            except Exception:
                arr = np.array([float(v) if v not in (None, "", "nan", "NaN") else np.nan for v in arr.flat],
                               dtype=np.float32).reshape(arr.shape)
        else:
            arr = arr.astype(np.float32)
        return arr

    def make_sequences(X, y, seq_len=24):
        X = ensure_numeric_array(X)
        y = ensure_numeric_array(y).reshape(-1)
        n = X.shape[0]
        if n < seq_len:
            raise ValueError(f"Not enough samples ({n}) for seq_len={seq_len}")
        n_windows = n - seq_len + 1
        n_features = X.shape[1]
        Xs = np.empty((n_windows, seq_len, n_features), dtype=np.float32)
        for i in range(n_windows):
            Xs[i] = X[i:i + seq_len]
        ys = y[seq_len - 1:]
        return Xs, ys

    # --- convert / build sequences before fit ---
    SEQ_LEN = 24  # set to whatever your LSTM expects
    X_train_seq, y_train_seq = make_sequences(X_train, y_train, seq_len=SEQ_LEN)
    X_val_seq, y_val_seq     = make_sequences(X_val, y_val, seq_len=SEQ_LEN)
    X_test_seq, y_test_seq   = make_sequences(X_test, y_test, seq_len=SEQ_LEN)

    # Debug prints & assertions
    print("Prepared for fit:")
    print("  X_train_seq.shape:", X_train_seq.shape, "dtype:", X_train_seq.dtype)
    print("  y_train_seq.shape:", y_train_seq.shape, "dtype:", y_train_seq.dtype)
    print("  X_val_seq.shape:", X_val_seq.shape, "dtype:", X_val_seq.dtype)
    assert X_train_seq.dtype == np.float32
    assert y_train_seq.dtype == np.float32
    assert X_train_seq.shape[0] == y_train_seq.shape[0]

    # Now pass the sequence arrays to model.fit
    # e.g. model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), ...)
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=...,
        batch_size=...
    )
