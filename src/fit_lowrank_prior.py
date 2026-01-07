"""
Standalone script for fitting the low-rank feature-based prior.

This script fits A = X @ W_A, B = W_B @ X.T where G = AB,
and saves all outputs for use in the main IBCD pipeline.

Usage:
    python fit_lowrank_prior.py \
        --features gene_features.csv \
        --r_hat R.csv \
        --se_hat SE_hat.csv \
        --output_dir prior_output/ \
        --rank 10 \
        --fit_mode regression \
        --lambda_reg 0.1
"""

import os
import argparse
import numpy as np
import pandas as pd

from feature_prior import (
    load_gene_features,
    fit_lowrank_regression,
    fit_lowrank_classification,
    compute_lowrank_sparsity,
    find_scale_for_sparsity,
)
from empirical_prior import load_R_and_SE_hat, empirical_bayes_em


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    X_features = load_gene_features(args.features)
    R_hat_df = pd.read_csv(args.r_hat)
    R_hat = R_hat_df.values
    colnames = R_hat_df.columns.tolist()

    SE_hat = None
    if args.se_hat:
        SE_hat = pd.read_csv(args.se_hat).values

    D, F = X_features.shape
    K = args.rank

    print(f"Gene features: {D} genes x {F} features")
    print(f"Rank K = {K}")
    print(f"Fit mode: {args.fit_mode}")
    print(f"Lambda regularization: {args.lambda_reg}")

    # Fit model
    if args.fit_mode == "regression":
        print("\nFitting low-rank model (regression mode)...")
        W_A, W_B, A_fitted, B_fitted = fit_lowrank_regression(
            X_features,
            R_hat,
            K=K,
            lambda_reg=args.lambda_reg,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
        )
    else:
        if SE_hat is None:
            raise ValueError("--se_hat is required for classification mode")
        print("\nFitting low-rank model (classification mode)...")
        W_A, W_B, A_fitted, B_fitted = fit_lowrank_classification(
            X_features,
            R_hat,
            SE_hat,
            K=K,
            lambda_reg=args.lambda_reg,
            threshold=args.threshold,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
        )

    # Compute predictions
    # G_pred = AB is our estimate of the direct effects matrix G
    # Note: G ≈ R - I (linear approximation), so G_pred should have ~0 diagonal
    G_pred = A_fitted @ B_fitted

    # Compute sparsity priors
    # Use EM to learn target sparsity if scale not provided
    if args.sparsity_scale is None:
        print("\nLearning sparsity level via EM...")
        w, se_flat = load_R_and_SE_hat(args.r_hat, args.se_hat if args.se_hat else args.r_hat)
        pi0_global, _, _ = empirical_bayes_em(w, se_flat)
        target_sparsity = float(pi0_global)
        print(f"  EM estimated global sparsity: {target_sparsity:.3f}")
        scale_used = find_scale_for_sparsity(A_fitted, B_fitted, target_sparsity)
        print(f"  Computed scale to match: {scale_used:.3f}")
        pi0_A, pi0_B = compute_lowrank_sparsity(A_fitted, B_fitted, target_sparsity=target_sparsity)
    else:
        scale_used = args.sparsity_scale
        pi0_A, pi0_B = compute_lowrank_sparsity(A_fitted, B_fitted, scale=scale_used)

    # Save outputs
    print(f"\nSaving outputs to {args.output_dir}/...")

    # Feature weights (for interpretability)
    feature_names = None
    if args.feature_names:
        feature_df = pd.read_csv(args.features)
        # Try to get feature names from columns
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_names = numeric_cols

    if feature_names and len(feature_names) == F:
        pd.DataFrame(W_A, index=feature_names, columns=[f"K{k}" for k in range(K)]).to_csv(
            f"{args.output_dir}/W_A.csv", index=True
        )
        pd.DataFrame(W_B, index=[f"K{k}" for k in range(K)], columns=feature_names).to_csv(
            f"{args.output_dir}/W_B.csv", index=True
        )
    else:
        pd.DataFrame(W_A, columns=[f"K{k}" for k in range(K)]).to_csv(
            f"{args.output_dir}/W_A.csv", index=False
        )
        pd.DataFrame(W_B, columns=[f"F{f}" for f in range(F)]).to_csv(
            f"{args.output_dir}/W_B.csv", index=False
        )

    # Fitted factor matrices
    pd.DataFrame(A_fitted, index=colnames, columns=[f"K{k}" for k in range(K)]).to_csv(
        f"{args.output_dir}/A_fitted.csv", index=True
    )
    pd.DataFrame(B_fitted, index=[f"K{k}" for k in range(K)], columns=colnames).to_csv(
        f"{args.output_dir}/B_fitted.csv", index=True
    )

    # Predicted G matrix
    pd.DataFrame(G_pred, index=colnames, columns=colnames).to_csv(
        f"{args.output_dir}/G_pred.csv", index=True
    )

    # Sparsity priors
    pd.DataFrame(pi0_A, index=colnames, columns=[f"K{k}" for k in range(K)]).to_csv(
        f"{args.output_dir}/pi0_A.csv", index=True
    )
    pd.DataFrame(pi0_B, index=[f"K{k}" for k in range(K)], columns=colnames).to_csv(
        f"{args.output_dir}/pi0_B.csv", index=True
    )

    # Save numpy arrays for direct loading
    np.save(f"{args.output_dir}/W_A.npy", W_A)
    np.save(f"{args.output_dir}/W_B.npy", W_B)
    np.save(f"{args.output_dir}/A_fitted.npy", A_fitted)
    np.save(f"{args.output_dir}/B_fitted.npy", B_fitted)
    np.save(f"{args.output_dir}/G_pred.npy", G_pred)
    np.save(f"{args.output_dir}/pi0_A.npy", pi0_A)
    np.save(f"{args.output_dir}/pi0_B.npy", pi0_B)

    # Save metadata
    metadata = {
        "D": D,
        "F": F,
        "K": K,
        "fit_mode": args.fit_mode,
        "lambda_reg": args.lambda_reg,
        "sparsity_scale": scale_used,
        "sparsity_learned_via_em": args.sparsity_scale is None,
    }
    pd.Series(metadata).to_csv(f"{args.output_dir}/metadata.csv")

    print("\nOutputs saved:")
    print(f"  - W_A.csv, W_A.npy: Feature weights for A ({F} x {K})")
    print(f"  - W_B.csv, W_B.npy: Feature weights for B ({K} x {F})")
    print(f"  - A_fitted.csv, A_fitted.npy: Fitted A matrix ({D} x {K})")
    print(f"  - B_fitted.csv, B_fitted.npy: Fitted B matrix ({K} x {D})")
    print(f"  - G_pred.csv, G_pred.npy: Predicted G = AB ({D} x {D})")
    print(f"  - pi0_A.csv, pi0_A.npy: Sparsity prior for A ({D} x {K})")
    print(f"  - pi0_B.csv, pi0_B.npy: Sparsity prior for B ({K} x {D})")
    print(f"  - metadata.csv: Fitting metadata")

    # Print summary statistics
    print("\nSummary:")
    print(f"  G_pred range: [{G_pred.min():.4f}, {G_pred.max():.4f}]")
    print(f"  G_pred mean (off-diag): {G_pred[~np.eye(D, dtype=bool)].mean():.4f}")
    print(f"  G_pred diagonal mean: {np.diag(G_pred).mean():.4f} (should be ~0)")
    print(f"  pi0_A mean: {pi0_A.mean():.4f} (sparsity level)")
    print(f"  pi0_B mean: {pi0_B.mean():.4f} (sparsity level)")

    # Reconstruction error vs target (R_hat - I ≈ G)
    # Linear approximation: R = (I-G)^{-1} ≈ I + G, so G ≈ R - I
    target = R_hat - np.eye(D)
    mask = ~np.eye(D, dtype=bool)
    mse_offdiag = ((target - G_pred)[mask] ** 2).mean()
    print(f"  MSE(R_hat - I, G_pred) off-diag: {mse_offdiag:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit low-rank feature-based prior for IBCD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python fit_lowrank_prior.py \\
        --features gene_features.csv \\
        --r_hat R.csv \\
        --se_hat SE_hat.csv \\
        --output_dir prior_output/ \\
        --rank 10 \\
        --fit_mode regression
        """
    )

    parser.add_argument(
        "--features",
        required=True,
        help="Path to gene features CSV (D genes x F features).",
    )

    parser.add_argument(
        "--r_hat",
        required=True,
        help="Path to R_hat CSV (D x D) from IV regression.",
    )

    parser.add_argument(
        "--se_hat",
        default=None,
        help="Path to SE_hat CSV (D x D). Required for classification mode.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save all outputs.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=10,
        help="Rank K for low-rank factorization G=AB. Default = 10.",
    )

    parser.add_argument(
        "--fit_mode",
        choices=["regression", "classification"],
        default="regression",
        help="Fitting mode: 'regression' fits to R_hat, 'classification' fits to significant edges. Default = regression.",
    )

    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.1,
        help="L1 regularization strength. Default = 0.1.",
    )

    parser.add_argument(
        "--sparsity_scale",
        type=float,
        default=None,
        help="Scale factor for converting fitted A,B to sparsity priors. If not set, uses EM to learn sparsity level from data.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for classification mode. Default = 2.0.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization. Default = 0.01.",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=2000,
        help="Maximum optimization iterations. Default = 2000.",
    )

    parser.add_argument(
        "--feature_names",
        action="store_true",
        help="Try to extract feature names from the features CSV.",
    )

    args = parser.parse_args()
    main(args)
