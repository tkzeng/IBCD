"""
Low-rank MCMC model for IBCD.

This module defines a NumPyro model where G = AB with spike-and-slab
horseshoe priors on the factor matrices A and B.
"""

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import numpy as np


def matrix_model_lowrank(
    obs_data,
    pi0_A,
    pi0_B,
    U_lower,
    V_lower,
    D,
    K,
    sigma0=0.001,
    tau=0.1,
    epsilon=1e-5
):
    """
    NumPyro model: Low-rank G = AB with spike-and-slab horseshoe priors.

    Model:
        A[i,k] ~ pi0_A[i,k] * N(0, sigma0^2) + (1-pi0_A[i,k]) * Horseshoe(tau)
        B[k,j] ~ pi0_B[k,j] * N(0, sigma0^2) + (1-pi0_B[k,j]) * Horseshoe(tau)
        G = A @ B
        R = (I - G)^{-1}
        R_hat ~ MatrixNormal(R, U, V)

    Parameters:
        obs_data: (D, D) observed R_hat matrix
        pi0_A: (D, K) spike probabilities for A
        pi0_B: (K, D) spike probabilities for B
        U_lower: (D, D) Cholesky of row covariance
        V_lower: (D, D) Cholesky of column covariance
        D: number of genes
        K: rank of factorization
        sigma0: standard deviation for spike component
        tau: global scale for horseshoe slab
        epsilon: regularization for numerical stability
    """
    # ===== Factor A (D x K) =====
    # Horseshoe local scales for A
    lam_A = numpyro.sample(
        "lam_A",
        dist.HalfCauchy(1.0).expand((D, K)),
        infer={"is_auxiliary": True}
    )

    # Standard normal for non-centered parameterization
    eps_A = numpyro.sample(
        "eps_A",
        dist.Normal(0.0, 1.0).expand((D, K)),
        infer={"is_auxiliary": True}
    )

    # Slab values (horseshoe)
    slab_A = tau * lam_A * eps_A

    # Spike values (near-zero)
    spike_A = numpyro.sample(
        "spike_A",
        dist.Normal(0.0, sigma0).expand((D, K)),
        infer={"is_auxiliary": True}
    )

    # Mixture: A = pi0_A * spike + (1 - pi0_A) * slab
    A = pi0_A * spike_A + (1.0 - pi0_A) * slab_A
    numpyro.deterministic("A", A)

    # ===== Factor B (K x D) =====
    # Horseshoe local scales for B
    lam_B = numpyro.sample(
        "lam_B",
        dist.HalfCauchy(1.0).expand((K, D)),
        infer={"is_auxiliary": True}
    )

    # Standard normal for non-centered parameterization
    eps_B = numpyro.sample(
        "eps_B",
        dist.Normal(0.0, 1.0).expand((K, D)),
        infer={"is_auxiliary": True}
    )

    # Slab values (horseshoe)
    slab_B = tau * lam_B * eps_B

    # Spike values (near-zero)
    spike_B = numpyro.sample(
        "spike_B",
        dist.Normal(0.0, sigma0).expand((K, D)),
        infer={"is_auxiliary": True}
    )

    # Mixture: B = pi0_B * spike + (1 - pi0_B) * slab
    B = pi0_B * spike_B + (1.0 - pi0_B) * slab_B
    numpyro.deterministic("B", B)

    # ===== G = A @ B =====
    G = A @ B
    numpyro.deterministic("G", G)

    # ===== Likelihood =====
    # R = (I - G)^{-1}
    I = jnp.eye(D)
    I_minus_G = I - G + epsilon * I  # regularization for stability
    R_mean = jnp.linalg.solve(I_minus_G, I)

    # MatrixNormal likelihood
    numpyro.sample(
        "R_hat_obs",
        dist.MatrixNormal(
            loc=R_mean,
            scale_tril_row=U_lower[:D, :D],
            scale_tril_column=V_lower[:D, :D],
        ),
        obs=obs_data[:D, :D],
    )


def matrix_model_lowrank_simple(
    obs_data,
    pi0_A,
    pi0_B,
    U_lower,
    V_lower,
    D,
    K,
    sigma0=0.001,
    tau=0.1
):
    """
    Simplified low-rank model without matrix inverse (for faster inference).

    Uses linear approximation: R ≈ I + G = I + AB

    This is faster but less accurate for large effects.

    Parameters: Same as matrix_model_lowrank
    """
    # ===== Factor A (D x K) =====
    lam_A = numpyro.sample(
        "lam_A",
        dist.HalfCauchy(1.0).expand((D, K)),
        infer={"is_auxiliary": True}
    )
    eps_A = numpyro.sample(
        "eps_A",
        dist.Normal(0.0, 1.0).expand((D, K)),
        infer={"is_auxiliary": True}
    )
    slab_A = tau * lam_A * eps_A
    spike_A = numpyro.sample(
        "spike_A",
        dist.Normal(0.0, sigma0).expand((D, K)),
        infer={"is_auxiliary": True}
    )
    A = pi0_A * spike_A + (1.0 - pi0_A) * slab_A
    numpyro.deterministic("A", A)

    # ===== Factor B (K x D) =====
    lam_B = numpyro.sample(
        "lam_B",
        dist.HalfCauchy(1.0).expand((K, D)),
        infer={"is_auxiliary": True}
    )
    eps_B = numpyro.sample(
        "eps_B",
        dist.Normal(0.0, 1.0).expand((K, D)),
        infer={"is_auxiliary": True}
    )
    slab_B = tau * lam_B * eps_B
    spike_B = numpyro.sample(
        "spike_B",
        dist.Normal(0.0, sigma0).expand((K, D)),
        infer={"is_auxiliary": True}
    )
    B = pi0_B * spike_B + (1.0 - pi0_B) * slab_B
    numpyro.deterministic("B", B)

    # ===== G = A @ B =====
    G = A @ B
    numpyro.deterministic("G", G)

    # ===== Linear approximation: R ≈ I + G =====
    R_mean = jnp.eye(D) + G

    # MatrixNormal likelihood
    numpyro.sample(
        "R_hat_obs",
        dist.MatrixNormal(
            loc=R_mean,
            scale_tril_row=U_lower[:D, :D],
            scale_tril_column=V_lower[:D, :D],
        ),
        obs=obs_data[:D, :D],
    )
