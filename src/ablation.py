import os
import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import arviz as az
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.nn import softmax
from scipy.optimize import minimize
from jax.nn import softmax
from scipy.optimize import minimize
import seaborn as sns
from tqdm import tqdm
import cvxpy as cp
import numpyro
import numpyro.distributions as dist
import arviz as az
import pandas as pd
import jax.scipy.special
from jax.scipy.stats import norm
from scipy.optimize import minimize

def softmax(x):
    """Stable softmax function using JAX."""
    return jax.nn.softmax(x)

def negative_log_likelihood_slab(theta, data, sigma_array, pi0, alpha=1.0):
    """
    Negative log-likelihood for the slab components, fixing the spike component pi0.
    """
    slab_pis = softmax(theta)  # shape (K,)
    slab_pis = (1 - pi0) * slab_pis  # Ensure the sum of probabilities = 1
    slab_sigmas = sigma_array

    data = jnp.array(data)
    nonzero_mask = (data != 0)
    x_nonzero = data[nonzero_mask]

    # Compute log-likelihood for nonzero data using the mixture model
    log_components = []
    for k in range(len(slab_sigmas)):
        log_components.append(
            jnp.log(slab_pis[k] + 1e-30) + norm.logpdf(x_nonzero, 0.0, slab_sigmas[k])
        )
    log_components = jnp.stack(log_components, axis=0)  # shape (K, N_nonzero)
    ll_nonzero = jnp.sum(jax.scipy.special.logsumexp(log_components, axis=0))

    # Add penalty (Dirichlet-like)
    penalty = (alpha - 1.0) * jnp.sum(jnp.log(slab_pis + 1e-30))

    return -(ll_nonzero + penalty)

def optimize_slab_weights(data, pi0, sigma_min=0.1, sigma_max=5.0, num_slabs=50, alpha=1.0):
    """
    Optimize the weights of the slab components while keeping pi0 fixed.
    """
    slab_scales = jnp.linspace(sigma_min, sigma_max, num_slabs)
    sigma_array = slab_scales  # No dummy sigma[0] since pi0 is fixed

    init_theta = np.zeros(num_slabs)  # Uniform initialization

    def loss_fn_for_scipy(theta_np):
        return negative_log_likelihood_slab(
            theta_np, data, sigma_array, pi0, alpha=alpha
        )

    grad_fn = jax.grad(negative_log_likelihood_slab, argnums=0)

    def grad_for_scipy(theta_np):
        g = grad_fn(
            jnp.array(theta_np),
            data,
            sigma_array,
            pi0,
            alpha
        )
        return np.array(g, dtype=np.float64)

    # Minimize
    result = minimize(
        fun=loss_fn_for_scipy,
        x0=init_theta,
        jac=grad_for_scipy,
        method='BFGS',
        options={'disp': True, 'maxiter': 500}
    )
    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    theta_opt = jnp.array(result.x)
    pi_slabs = softmax(theta_opt)  # shape (K,)
    pi_slabs = (1 - pi0) * pi_slabs  # Normalize to sum to 1 - pi0

    return np.array(pi_slabs), np.array(slab_scales)

def low_rank_approximation(U, singular_values, Vt, rank):
    # Use the top 'rank' singular values and vectors
    S_approx = np.dot(U[:, :rank], np.dot(np.diag(singular_values[:rank]), Vt[:rank, :]))
    return S_approx
    

def mvn_model_spike_horseshoe(S_jax, obs_data, pi0_ij, covariance_matrix, D, sigma0=0.001, tau=0.1, epsilon=1e-5):
    """
    NumPyro model: Spike-and-horseshoe prior over matrix G, MatrixNormal likelihood.

    Parameters:
        obs_data (array): R_hat matrix.
        pi0_ij (array): Edge-specific spike weights.
        U_lower (array): Cholesky of row covariance.
        V_lower (array): Cholesky of column covariance.
        D (int): Dimension.
        sigma0 (float): Std for spike component.
        tau (float): Global scale for horseshoe slab.
        epsilon (float): Regularization for stability.
    """

    # Sample horseshoe local scales (HalfCauchy), shape (D, D)
    lam = numpyro.sample("lam", dist.HalfCauchy(1.).expand((D, D)), infer={"is_auxiliary": True})

    # Sample standard normal noise, for non-centered parameterization
    eps = numpyro.sample("eps", dist.Normal(0., 1.).expand((D, D)), infer={"is_auxiliary": True})

    # Slab values: non-centered horseshoe (continuous)
    slab_vals = tau * lam * eps

    # Spike values: zero-mean Normal with tiny variance
    spike_vals = numpyro.sample("spike", dist.Normal(0., sigma0).expand((D, D)), infer={"is_auxiliary": True})

    # Binary mixture via convex combination (no discrete Cat)
    G = pi0_ij * spike_vals + (1. - pi0_ij) * slab_vals
    numpyro.deterministic("G", G)           # keep only G

    # MatrixNormal mean
    I = jnp.eye(D)
    I_minus_G = I - G + epsilon * jnp.eye(D)
    R_mean = jnp.linalg.solve(I_minus_G, I)

    # MatrixNormal likelihood
    numpyro.sample(
        'R_hat_obs',
        dist.MultivariateNormal(
            loc=R_mean,
            covariance_matrix=S_jax[:D, :D]  # Use only a D x D submatrix
        ),
        obs=obs_data[:D, :D]
    )

def is_positive_definite(matrix, tol=0):
    eigenvalues = np.linalg.eigvalsh(np.array(matrix))  # Convert to NumPy for eigvalsh
    return np.all(eigenvalues > tol)

def make_positive_definite(matrix, jitter=1e-5):
    """
    Ensures a matrix is positive definite by:
    1. Symmetrizing it.
    2. Flipping negative eigenvalues.
    3. Adding jitter to the diagonal.
    """
    matrix = (matrix + matrix.T) / 2  
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.array(matrix))  # Convert to NumPy for eigendecomposition
    # Flip negative eigenvalues to positive
    eigvals[eigvals < jitter] = jitter  

    # Reconstruct the matrix
    matrix_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Convert back to JAX array
    return jnp.array(matrix_fixed)