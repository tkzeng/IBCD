"""
Feature-based low-rank prior fitting for IBCD.

This module fits A = X @ W_A, B = W_B @ X.T where G = AB,
using gene features to parameterize the low-rank factorization.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad
import optax
from scipy.special import expit
from tqdm import tqdm


def load_gene_features(feature_path: str) -> np.ndarray:
    """
    Load gene feature matrix from CSV.

    Parameters:
        feature_path: Path to CSV file with shape (D genes x F features).
                     First column can be gene names (will be dropped if non-numeric).

    Returns:
        X_features: numpy array of shape (D, F)
    """
    df = pd.read_csv(feature_path)

    # Drop non-numeric columns (e.g., gene name index)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X_features = df[numeric_cols].values.astype(np.float64)

    # Standardize features
    X_mean = X_features.mean(axis=0, keepdims=True)
    X_std = X_features.std(axis=0, keepdims=True) + 1e-8
    X_features = (X_features - X_mean) / X_std

    return X_features


def _predict_G(W_A, W_B, X_features):
    """
    Compute G = A @ B where A = X @ W_A, B = W_B @ X.T

    Parameters:
        W_A: (F, K) weight matrix for A
        W_B: (K, F) weight matrix for B
        X_features: (D, F) gene feature matrix

    Returns:
        G: (D, D) predicted effect matrix
        A: (D, K) factor matrix
        B: (K, D) factor matrix
    """
    A = X_features @ W_A  # D x K
    B = W_B @ X_features.T  # K x D
    G = A @ B  # D x D
    return G, A, B


def _loss_regression(params, X_features, target, mask, lambda_reg):
    """
    Regression loss: ||target - G||^2 + lambda * (||W_A||_1 + ||W_B||_1)

    Linear approximation note:
        R = (I - G)^{-1} ≈ I + G  (for small G)
        Therefore: G ≈ R - I

        We fit G to (R_hat - I), which estimates total effects.
        The diagonal is masked since G_ii = 0 (no self-loops).

    Parameters:
        params: dict with 'W_A' and 'W_B'
        X_features: (D, F) gene features
        target: (D, D) target matrix, should be (R_hat - I)
        mask: (D, D) boolean mask for off-diagonal entries
        lambda_reg: L1 regularization strength

    Returns:
        loss: scalar loss value
    """
    W_A, W_B = params['W_A'], params['W_B']
    G_pred, _, _ = _predict_G(W_A, W_B, X_features)

    # MSE on off-diagonal entries only
    # G diagonal should be 0 (no self-loops), R diagonal is 1
    residual = (G_pred - target) ** 2
    mse = residual[mask].mean()

    # L1 regularization for sparsity
    reg = lambda_reg * (jnp.abs(W_A).sum() + jnp.abs(W_B).sum())

    return mse + reg


def _loss_classification(params, X_features, labels, mask, lambda_reg):
    """
    Classification loss: BCE(sigmoid(G), labels) + lambda * (||W_A||_1 + ||W_B||_1)

    Note: Labels are based on significance of (R_hat - I), which approximates G.
    Diagonal is masked since G_ii = 0 (no self-loops).

    Parameters:
        params: dict with 'W_A' and 'W_B'
        X_features: (D, F) gene features
        labels: (D, D) binary labels (1 = significant edge in G)
        mask: (D, D) boolean mask for off-diagonal entries
        lambda_reg: L1 regularization strength

    Returns:
        loss: scalar loss value
    """
    W_A, W_B = params['W_A'], params['W_B']
    G_pred, _, _ = _predict_G(W_A, W_B, X_features)

    # Binary cross-entropy on off-diagonal only (takes logits directly)
    bce = optax.losses.sigmoid_binary_cross_entropy(G_pred, labels)
    loss = bce[mask].mean()

    # L1 regularization
    reg = lambda_reg * (jnp.abs(W_A).sum() + jnp.abs(W_B).sum())

    return loss + reg


def fit_lowrank_regression(
    X_features: np.ndarray,
    R_hat: np.ndarray,
    K: int = 10,
    lambda_reg: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 2000,
    tol: float = 1e-6
):
    """
    Fit A = X @ W_A, B = W_B @ X.T to predict G via regression.

    Linear approximation:
        R = (I - G)^{-1} ≈ I + G  (for small G)
        Therefore: G ≈ R - I

    We fit G = AB to (R_hat - I) on off-diagonal entries only.
    Diagonal is masked since G_ii = 0 (no self-loops).

    Loss: ||(R_hat - I) - AB||^2 + lambda * (||W_A||_1 + ||W_B||_1)
          (off-diagonal only)

    Parameters:
        X_features: (D, F) gene feature matrix
        R_hat: (D, D) estimated total effects from IV regression
        K: rank of factorization
        lambda_reg: L1 regularization strength
        learning_rate: optimizer learning rate
        max_iter: maximum optimization iterations
        tol: convergence tolerance

    Returns:
        W_A: (F, K) fitted weight matrix
        W_B: (K, F) fitted weight matrix
        A_fitted: (D, K) fitted factor matrix
        B_fitted: (K, D) fitted factor matrix
    """
    D, F = X_features.shape
    X_features = jnp.array(X_features)

    # Target is G ≈ R - I (linear approximation)
    # Diagonal of target is 0 (will be masked anyway)
    target = jnp.array(R_hat) - jnp.eye(D)

    # Off-diagonal mask (G_ii = 0)
    mask = ~jnp.eye(D, dtype=bool)

    # Initialize weights (small random values)
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    params = {
        'W_A': jax.random.normal(key1, (F, K)) * 0.01,
        'W_B': jax.random.normal(key2, (K, F)) * 0.01,
    }

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # JIT compile gradient function
    loss_fn = lambda p: _loss_regression(p, X_features, target, mask, lambda_reg)
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_fn_jit = jax.jit(loss_fn)

    # Optimization loop
    prev_loss = float('inf')
    pbar = tqdm(range(max_iter), desc="Fitting low-rank (regression)")

    for i in pbar:
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if i % 100 == 0:
            loss = float(loss_fn_jit(params))
            pbar.set_postfix({'loss': f'{loss:.6f}'})

            if abs(prev_loss - loss) < tol:
                print(f"Converged at iteration {i}")
                break
            prev_loss = loss

    # Extract results
    W_A = np.array(params['W_A'])
    W_B = np.array(params['W_B'])
    A_fitted = X_features @ params['W_A']
    B_fitted = params['W_B'] @ X_features.T

    return W_A, W_B, np.array(A_fitted), np.array(B_fitted)


def fit_lowrank_classification(
    X_features: np.ndarray,
    R_hat: np.ndarray,
    SE_hat: np.ndarray,
    K: int = 10,
    lambda_reg: float = 0.1,
    threshold: float = 2.0,
    learning_rate: float = 0.01,
    max_iter: int = 2000,
    tol: float = 1e-6
):
    """
    Fit A = X @ W_A, B = W_B @ X.T to classify significant edges in G.

    Linear approximation context:
        R = (I - G)^{-1} ≈ I + G  (for small G)
        Therefore: G ≈ R - I

    Labels are based on |(R_hat - I) / SE_hat| > threshold, which identifies
    significant entries in G (off-diagonal only, since G_ii = 0).

    Loss: BCE(sigmoid(AB), labels) + lambda * (||W_A||_1 + ||W_B||_1)
          (off-diagonal only)

    Parameters:
        X_features: (D, F) gene feature matrix
        R_hat: (D, D) estimated total effects
        SE_hat: (D, D) standard errors
        K: rank of factorization
        lambda_reg: L1 regularization strength
        threshold: z-score threshold for significant edges
        learning_rate: optimizer learning rate
        max_iter: maximum optimization iterations
        tol: convergence tolerance

    Returns:
        W_A: (F, K) fitted weight matrix
        W_B: (K, F) fitted weight matrix
        A_fitted: (D, K) fitted factor matrix
        B_fitted: (K, D) fitted factor matrix
    """
    D, F = X_features.shape
    X_features = jnp.array(X_features)

    # G ≈ R - I (linear approximation)
    G_approx = R_hat - np.eye(D)

    # Create binary labels based on z-score threshold for G (off-diagonal)
    # Note: SE_hat is for R, using as proxy for G uncertainty
    z_scores = np.abs(G_approx) / (SE_hat + 1e-8)
    labels = jnp.array((z_scores > threshold).astype(np.float32))

    # Off-diagonal mask (G_ii = 0)
    mask = ~jnp.eye(D, dtype=bool)

    # Initialize weights
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    params = {
        'W_A': jax.random.normal(key1, (F, K)) * 0.01,
        'W_B': jax.random.normal(key2, (K, F)) * 0.01,
    }

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # JIT compile
    loss_fn = lambda p: _loss_classification(p, X_features, labels, mask, lambda_reg)
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_fn_jit = jax.jit(loss_fn)

    # Optimization loop
    prev_loss = float('inf')
    pbar = tqdm(range(max_iter), desc="Fitting low-rank (classification)")

    for i in pbar:
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if i % 100 == 0:
            loss = float(loss_fn_jit(params))
            pbar.set_postfix({'loss': f'{loss:.6f}'})

            if abs(prev_loss - loss) < tol:
                print(f"Converged at iteration {i}")
                break
            prev_loss = loss

    # Extract results
    W_A = np.array(params['W_A'])
    W_B = np.array(params['W_B'])
    A_fitted = X_features @ params['W_A']
    B_fitted = params['W_B'] @ X_features.T

    return W_A, W_B, np.array(A_fitted), np.array(B_fitted)


def find_scale_for_sparsity(
    A_fitted: np.ndarray,
    B_fitted: np.ndarray,
    target_sparsity: float
) -> float:
    """
    Find scale such that average sparsity (pi0) matches target.

    Uses bisection search to find scale where:
        mean(pi0_A) ≈ target_sparsity
        mean(pi0_B) ≈ target_sparsity

    Parameters:
        A_fitted: (D, K) fitted factor matrix
        B_fitted: (K, D) fitted factor matrix
        target_sparsity: desired average pi0 (e.g., 0.8 for 80% sparse)

    Returns:
        scale: float that achieves target sparsity
    """
    from scipy.optimize import brentq

    def avg_sparsity(scale):
        pi0_A = expit(-np.abs(A_fitted) * scale)
        pi0_B = expit(-np.abs(B_fitted) * scale)
        return (pi0_A.mean() + pi0_B.mean()) / 2

    def objective(scale):
        return avg_sparsity(scale) - target_sparsity

    # Check bounds
    sparsity_low = avg_sparsity(0.01)   # low scale = sparsity near 0.5
    sparsity_high = avg_sparsity(100.0)  # high scale = sparsity near 1.0 or 0.0

    # If target is out of range, return boundary
    if target_sparsity >= sparsity_low:
        return 0.01
    if target_sparsity <= sparsity_high:
        return 100.0

    try:
        scale = brentq(objective, 0.01, 100.0)
    except ValueError:
        # Fallback if brentq fails
        scale = 10.0

    return scale


def compute_lowrank_sparsity(
    A_fitted: np.ndarray,
    B_fitted: np.ndarray,
    scale: float = None,
    target_sparsity: float = None
):
    """
    Convert fitted |A|, |B| to sparsity priors.

    pi0_A = sigmoid(-|A_fitted| * scale)
    pi0_B = sigmoid(-|B_fitted| * scale)

    Small fitted values -> high pi0 (more likely spike/zero)
    Large fitted values -> low pi0 (more likely slab/non-zero)

    Parameters:
        A_fitted: (D, K) fitted factor matrix
        B_fitted: (K, D) fitted factor matrix
        scale: scaling factor for sigmoid (larger = sharper transition).
               If None, must provide target_sparsity.
        target_sparsity: desired average pi0. If provided, scale is computed
                        automatically to match this target.

    Returns:
        pi0_A: (D, K) spike probabilities for A
        pi0_B: (K, D) spike probabilities for B
    """
    # Determine scale
    if scale is None and target_sparsity is None:
        scale = 10.0  # default
    elif scale is None:
        scale = find_scale_for_sparsity(A_fitted, B_fitted, target_sparsity)

    # Sigmoid transformation: small |A| -> pi0 near 1, large |A| -> pi0 near 0
    pi0_A = expit(-np.abs(A_fitted) * scale)
    pi0_B = expit(-np.abs(B_fitted) * scale)

    # Clip to avoid numerical issues
    pi0_A = np.clip(pi0_A, 0.01, 0.99)
    pi0_B = np.clip(pi0_B, 0.01, 0.99)

    return pi0_A, pi0_B
