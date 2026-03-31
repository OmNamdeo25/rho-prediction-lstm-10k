#!/usr/bin/env python3
"""
part1_generate_dataset.py

Generate dataset for rho prediction using Gauss–Markov AR(1) model.
Input: sequence of channel vectors (real + imag)
Target: correlation coefficient rho (scalar)
No rho is included in the input features.
"""

import os
import numpy as np
import pandas as pd

SEED = 2025
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def generate_rho_dataset(n_samples=10000, n_antennas=4, rho_range=(0.6, 0.95),
                         sequence_length=10, noise_var=0.05, verbose=True):
    """
    Generate synthetic channel sequences and label each with its rho.
    
    Returns:
        X: (n_samples, sequence_length, 2*n_antennas)   # real + imag only
        y: (n_samples, 1)                               # rho value
        rho_values: (n_samples,)                       # same as y, for metadata
    """
    if verbose:
        print("="*60)
        print("GENERATING RHO PREDICTION DATASET")
        print(f"samples={n_samples}, antennas={n_antennas}, seq_len={sequence_length}")
        print(f"rho_range={rho_range}, noise_var={noise_var}")
        print("="*60)

    all_X = []
    all_y = []
    all_rho = []

    for i in range(n_samples):
        rho = np.random.uniform(rho_range[0], rho_range[1])

        # initial complex channel (unit variance)
        h = (np.random.randn(n_antennas) + 1j * np.random.randn(n_antennas)) / np.sqrt(2)

        seq = []
        for t in range(sequence_length):
            seq.append(h.copy())
            innovation_var = noise_var * (1 - rho**2)
            innovation = np.sqrt(max(innovation_var, 0.0)) * (
                np.random.randn(n_antennas) + 1j * np.random.randn(n_antennas)
            ) / np.sqrt(2)
            h = rho * h + innovation

        # convert each channel state to real features: [real parts, imag parts]
        features = []
        for h_t in seq:
            h_real = h_t.real
            h_imag = h_t.imag
            features.append(np.concatenate([h_real, h_imag]))   # no rho appended

        all_X.append(features)
        all_y.append([rho])
        all_rho.append(rho)

        if verbose and (i+1) % 1000 == 0:
            print(f"  Generated {i+1}/{n_samples} sequences...")

    X = np.array(all_X, dtype=np.float32)          # (n, seq_len, 2*n_ant)
    y = np.array(all_y, dtype=np.float32)          # (n, 1)
    rho_values = np.array(all_rho, dtype=np.float32)

    if verbose:
        print("Data generation complete.")
        print("  X shape:", X.shape)
        print("  y shape:", y.shape)
        print(f"  rho range: [{rho_values.min():.3f}, {rho_values.max():.3f}]")

    return X, y, rho_values


def save_rho_dataset_to_csv(X, y, rho_values, output_dir='./dataset', prefix='rho_data'):
    """Save dataset to CSV files (similar structure as original)."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*60)
    print("SAVING DATASET TO CSV")
    print("="*60)

    n_samples, seq_len, n_features = X.shape

    # Flatten X for CSV
    X_flattened = X.reshape(n_samples, -1)
    X_columns = [f"t{t}_f{f}" for t in range(seq_len) for f in range(n_features)]
    X_df = pd.DataFrame(X_flattened, columns=X_columns)
    X_path = os.path.join(output_dir, f'{prefix}_X.csv')
    X_df.to_csv(X_path, index=False)
    print(f"✓ Saved input to: {X_path}  shape: {X.shape}")

    # Save y (rho)
    y_df = pd.DataFrame(y, columns=['rho'])
    y_path = os.path.join(output_dir, f'{prefix}_y.csv')
    y_df.to_csv(y_path, index=False)
    print(f"✓ Saved targets to: {y_path}  shape: {y.shape}")

    # Metadata (sample id, rho again – for convenience)
    metadata_df = pd.DataFrame({'sample_id': np.arange(n_samples), 'rho': rho_values})
    metadata_path = os.path.join(output_dir, f'{prefix}_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Saved metadata to: {metadata_path}")

    # Shape info (essential for loading)
    shape_info = pd.DataFrame({
        'n_samples': [n_samples],
        'sequence_length': [seq_len],
        'n_features': [n_features],
        'output_dim': [y.shape[1]]
    })
    shape_path = os.path.join(output_dir, f'{prefix}_shape_info.csv')
    shape_df = pd.DataFrame(shape_info)
    shape_df.to_csv(shape_path, index=False)
    print(f"✓ Saved shape info to: {shape_path}")
    print("\n" + "="*60)
    print("DATASET SAVED SUCCESSFULLY!")
    print("="*60)


def main():
    print("\n" + "="*60)
    print(" PART 1: DATASET GENERATION FOR RHO PREDICTION")
    print(" Gauss-Markov Channel Model")
    print("="*60 + "\n")

    # ========= CONFIGURATION =========
    N_SAMPLES = 10000   # updated to 10,000 samples
    N_ANTENNAS = 4
    SEQ_LENGTH = 10
    RHO_RANGE = (0.6, 0.95)
    NOISE_VAR = 0.05
    OUTPUT_DIR = './dataset'
    PREFIX = 'rho_data'
    # ==================================

    print("Configuration:")
    print(f"  Samples: {N_SAMPLES}, Antennas: {N_ANTENNAS}, Seq length: {SEQ_LENGTH}")
    print(f"  Rho range: {RHO_RANGE}, Noise var: {NOISE_VAR}")
    print(f"  Output: {OUTPUT_DIR}/{PREFIX}_*.csv\n")

    X, y, rho_vals = generate_rho_dataset(
        n_samples=N_SAMPLES,
        n_antennas=N_ANTENNAS,
        rho_range=RHO_RANGE,
        sequence_length=SEQ_LENGTH,
        noise_var=NOISE_VAR,
        verbose=True
    )

    save_rho_dataset_to_csv(X, y, rho_vals, output_dir=OUTPUT_DIR, prefix=PREFIX)

    print("\n✓ Dataset ready. You can now run part2_train_model.py")


if __name__ == "__main__":
    main()