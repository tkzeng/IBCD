import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import arviz as az
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import config
config.update("jax_enable_x64", True)
import cvxpy as cp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS



def matrix_model_edge_specific(
    obs_data,        # Observed data matrix
    pi0_ij,          # (D,D) Edge-specific spike portion
    pi_k_ij,         # (D,D) Edge-specific total slab portion
    pi_k,            # (K,) Global slab portion across components
    sigma_grid,      # (K,) List of slab variances
    U_lower, V_lower, # Row/column covariances (Cholesky)
    D,               # Dimension
    sigma0=0.001,    # Spike std dev
    epsilon=1e-5
):
    """
    Edge-specific spike-and-slab prior for G_{ij}, where each edge (i,j)
    has mixture weights [ pi0_ij[i,j], pi_k_ij[i,j] * pi_k[1], ..., pi_k_ij[i,j] * pi_k[K] ].

    Summation over spike + slab = 1 for every (i,j).
    """

    # (1) Build per-edge mixture probabilities
    spike_probs = pi0_ij  # shape (D,D), edge-specific spike
    slab_probs_k = pi_k_ij[..., None] * pi_k[None, None, :]  # shape (D,D,K)

    # Combine along last axis => shape (D, D, K+1)
    mixing_probs = jnp.concatenate([spike_probs[..., None], slab_probs_k], axis=-1)

    # (2) Create categorical distribution for per-edge mixing
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # (3) Construct spike + K slab components, each expanded to (D,D)
    component_dists = []
    component_dists.append(dist.Normal(loc=0.0, scale=sigma0).expand((D, D)))  # Spike component
    for s_k in sigma_grid:
        component_dists.append(dist.Normal(loc=0.0, scale=s_k).expand((D, D)))  # Slab components

    # (4) Construct the full mixture prior
    G_prior = dist.MixtureGeneral(mixing_dist, component_dists)

    # Sample G from edge-specific mixture
    G = numpyro.sample("G", G_prior)

    # (5) Regularized inverse calc: (I - G + eps)
    I = jnp.eye(D)
    I_minus_G = I - G + epsilon * jnp.eye(D)
    R_mean = jnp.linalg.solve(I_minus_G, I)

    # (6) MatrixNormal likelihood
    numpyro.sample(
        "R_hat_obs",
        dist.MatrixNormal(loc=R_mean,
                          scale_tril_row=U_lower[:D, :D],
                          scale_tril_column=V_lower[:D, :D]),
        obs=obs_data[:D, :D]
    )
    
def plot_posterior_(idata,ax=None):
    #compare with
        #todo
    if ax is None:
        az.plot_posterior(idata.posterior.G)
        # plt.show()

def compute_lfsr(flat_samples):
    """
    Compute the Local False Sign Rate (LFSR) for each entry.
    LFSR_{ij} = min( P(G_{ij} > 0), P(G_{ij} < 0) ).
    
    Args:
        flat_samples: np.ndarray (total_samples, D, D)
    
    Returns:
        lfsr_matrix: (D, D) array in [0, 1]
    """
    # Probability each entry > 0 or < 0
    pos_probs = np.mean(flat_samples > 0, axis=0)
    neg_probs = np.mean(flat_samples < 0, axis=0)
    lfsr_matrix = np.minimum(pos_probs, neg_probs)
    return lfsr_matrix


def load_R_hat(filename):
    """
    Load the correlation matrix R_hat from CSV and extract upper-triangle off-diagonal values.
    
    Parameters:
        filename (str): Path to the CSV file containing R_hat.

    Returns:
        w (jnp.ndarray): Vector of off-diagonal values from the upper triangle of R_hat.
    """
    R_hat = pd.read_csv(filename).values  # Read CSV into numpy array
    N = R_hat.shape[0]
    
    # Extract upper-triangle (excluding diagonal) to avoid duplicate correlations
    w = R_hat[jnp.triu_indices(N, k=1)]  
    return w

def solve_spike_slab_diagonal_spike(
    xi,
    pi0,
    alpha_data=1.0,
    beta_global=1.0
):
    """
    We want:
      1) For off-diagonal (i != j): pi0_ij + pik_ij = 1, pi0_ij>=0, pik_ij>=0
      2) For diagonal (i == j): pi0_ii = 1,     pik_ii = 0
      3) The sum of off-diagonal pi0_ij ~ pi0 * (D^2 - D) (i.e. a global constraint)
      4) Also we 'data-fit' slab to some scaled xi_{ij} for off-diagonal.
    """

    D = xi.shape[0]

    pi0_var = cp.Variable((D,D), nonneg=True)
    pik_var = cp.Variable((D,D), nonneg=True)

    # constraints = []

    # # Mask for off-diagonal (i ≠ j)
    # offdiag_mask = np.ones((D, D), dtype=bool)
    # np.fill_diagonal(offdiag_mask, False)

    # # Enforce pi0 + pik == 1 only on off-diagonals
    # sum_matrix = pi0_var + pik_var
    # constraints.append(cp.multiply(offdiag_mask, sum_matrix) == 1)

    # Diagonal constraints explicitly
    # constraints.append(cp.diag(pi0_var) == np.ones(D))
    # constraints.append(cp.diag(pik_var) == np.zeros(D))

    constraints = []
    constraints = [pi0_var + pik_var == np.ones((D, D))]
    constraints += [cp.diag(pi0_var) == np.ones(D)]
    constraints += [cp.diag(pik_var) == np.zeros(D)]

    offdiag_mask = np.ones((D, D), dtype=bool)
    np.fill_diagonal(offdiag_mask, False)


    # A) For i != j: pi0_ij + pi_k_ij = 1
    #    For i == j: pi0_ii = 1 and pi_k_ii = 0

    # B) We'll "data-fit" the slab part to some scaled xi_{ij}, but only for off-diagonals
    #    so define a mask for i!=j
    # offdiag_mask = np.ones((D,D), dtype=bool)
    # np.fill_diagonal(offdiag_mask, False)

    # Normalize xi so that xi_{ij} is in [0,1] 
    # (or at least has maximum ~1 to define a "desired" slab)
    max_xi = xi.max()
    if max_xi < 1e-12:
        max_xi = 1e-12
    xi_norm = xi / max_xi

    # We'll build a sum-of-squares residual over i!=j
    #   "desired slab" = xi_norm, "desired spike" = 1 - xi_norm
    # but only for off-diagonal entries:
    data_resid = cp.sum_squares(
        cp.multiply(offdiag_mask, pik_var - xi_norm)
    ) + cp.sum_squares(
        cp.multiply(offdiag_mask, pi0_var - (1 - xi_norm))
    )

    # C) Global prior penalty: 
    #    We only want the sum over off-diagonal pi0_ij to be ~ pi0 * (D^2 - D).
    sum_pi0_offdiag = cp.sum(pi0_var) - cp.sum(cp.diag(pi0_var))
    #   i.e. pi0*(D^2 - D) is the target
    #global_penalty = cp.square(
    #    sum_pi0_offdiag - pi0*(D**2 - D)
    #)
    constraints.append(
        sum_pi0_offdiag == pi0*(D**2 - D)
    )
    # Weighted objective
    objective = alpha_data*data_resid # + beta_global*global_penalty

    # Solve
    prob = cp.Problem(cp.Minimize(objective), constraints)
    val = prob.solve()

    # Extract numeric solutions
    pi0_sol = pi0_var.value
    pik_sol = pik_var.value

    return pi0_sol, pik_sol, val


def empirical_bayes_em(w, K=50, sigma_min=0.01, sigma_max=1.0, alpha=2, max_iter=20000, tol=1e-6):
    """
    Run the EM algorithm for Empirical Bayes estimation of spike-and-slab mixture.

    Parameters:
        w (jnp.ndarray): Off-diagonal entries of R_hat (observations).
        K (int): Number of slab components.
        sigma_min (float): Minimum variance for slab components.
        sigma_max (float): Maximum variance for slab components.
        alpha (float): Penalty hyperparameter (default: 2).
        max_iter (int): Maximum number of EM iterations.
        tol (float): Convergence threshold for stopping criteria.

    Returns:
        pi_0 (float): Estimated proportion of the spike component.
        pi_k (jnp.ndarray): Estimated proportions of the slab components.
    """
    N = len(w)  # Number of data points (upper-triangle correlations)
    sigma_k = jnp.linspace(sigma_min, sigma_max, K)  # Grid of slab variances

    # Initialize mixture proportions
    pi_0 = jnp.array(0.5)  # Start with half in the spike component
    pi_k = jnp.full(K, (1 - pi_0) / K)  # Uniform initialization for slabs

    @jax.jit
    def em_step(params):
        """
        One iteration of the EM algorithm.
        
        Parameters:
            params (tuple): Current values of (pi_0, pi_k).
        
        Returns:
            tuple: Updated values of (pi_0, pi_k).
        """
        pi_0, pi_k = params  # Unpack current values

        # === E-Step: Compute responsibilities z_ik ===
        # Compute likelihoods for each mixture component using JAX's norm.pdf
        f_0 = norm.pdf(w, loc=0, scale=jnp.sqrt(0.001))  # Spike component (fixed variance)
        f_k = norm.pdf(w[:, None], loc=0, scale=jnp.sqrt(sigma_k))  # Slab components (K different variances)

        # Compute posterior probabilities (responsibilities)
        numerator_0 = pi_0 * f_0  # Weight * likelihood for spike
        numerator_k = pi_k * f_k  # Weight * likelihood for slabs (broadcasted over K)

        denominator = numerator_0 + numerator_k.sum(axis=1)  # Sum over all components

        # Compute responsibilities (posterior probabilities for each component)
        z_0 = numerator_0 / denominator  # Responsibility for the spike
        z_k = numerator_k / denominator[:, None]  # Responsibilities for each slab (shape: N x K)

        # Compute expected counts
        n_0 = jnp.sum(z_0)  # Expected number of points in the spike component
        n_k = jnp.sum(z_k, axis=0)  # Expected number of points in each slab component

        # === M-Step: Update pi_0 and pi_k using the penalty (alpha-1) on slabs only ===
        # total_counts = n_0 + jnp.sum(n_k + (alpha - 1))  # Correct denominator
        # pi_0_new = n_0 / total_counts  # Update spike weight (no penalty)
        # pi_k_new = (n_k + (alpha - 1)) / total_counts  # Update slab weights with penalty
   
        pi_0_new = n_0 / (N + (alpha - 1) * K)  # Update spike weight (no penalty)
        pi_k_new = (n_k + (alpha - 1)) / (N + (alpha - 1) * K)  # Update slab weights with penalty

        return pi_0_new, pi_k_new

    # === Iterate EM until convergence ===
    for i in tqdm(range(max_iter)):
        pi_0_new, pi_k_new = em_step((pi_0, pi_k))  # Perform one EM step

        # if i % 10 == 0:
        #     print(f"Iteration {i}: pi_0 = {pi_0_new:.4f}")
            
        # Check convergence (if change in pi_k is below threshold)
        if jnp.max(jnp.abs(pi_k_new - pi_k)) < tol:
            break

        # Update parameters for next iteration
        pi_0, pi_k = pi_0_new, pi_k_new

    return pi_0, pi_k, sigma_k


if __name__ == "__main__":
  path = "/Users/seongwoohan/Desktop/bcbg/ibcd/"
  print("Default device:", jax.devices()[0])

  G_df = pd.read_csv("G_matrix.csv")
  total_entries = G_df.size
  print(total_entries)
  zero_count = (G_df == 0).sum().sum()
  print(zero_count)
  count = zero_count / total_entries
  print(count)

  w = load_R_hat("R_500.csv")  

  print(w.shape)
  pi0, pi_slabs, slab_scales = empirical_bayes_em(w, alpha=2)

  # Print final estimated proportions
  print("Estimated pi_0 (spike weight):", pi0)
  print("Estimated pi_k (slab weights):", pi_slabs)
  print(f"Slab scales = {slab_scales[:]} ...")
  print(f"Check sum of all pi_k = {pi_slabs.sum():.4f}")
  print(f"Check sum of all pi = {pi0 + pi_slabs.sum():.4f}")

  xi_df = pd.read_csv("xi_500.csv", index_col=0)
  #xi_df = pd.read_csv("xi.csv", index_col=0)
  xi = xi_df.values
  D = xi.shape[0]

  # Load precomputed per-edge spike and slab weights
  pi0_ij, pi_k_ij, _ = solve_spike_slab_diagonal_spike(xi, pi0=pi0)
  print("pi0: ", pi0_ij)
  print("pik: ", pi_k_ij)

  df4 = pd.DataFrame(pi0_ij)
  df4.to_csv(path + "pi0_ij.csv", header=False, index=False)

  df5 = pd.DataFrame(pi_k_ij)
  df5.to_csv(path + "pi_k_ij.csv", header=False, index=False)

  pi0_ij  = jnp.full((D, D), pi0)       
  pi_k_ij = jnp.full((D, D), 1.0 - pi0)  
  print("pi0: ", pi0_ij)
  print("pik: ", pi_k_ij)

  # Load other required data
  Rhat_df = pd.read_csv("R_500.csv")
  U_df = pd.read_csv("U_500.csv")
  V_df = pd.read_csv("V_500.csv")

  # Convert to float and then JAX arrays
  U_lower = jnp.linalg.cholesky(jnp.array(U_df.values.astype(float)))
  V_lower = jnp.linalg.cholesky(jnp.array(V_df.values.astype(float)))

  # MCMC setup
  kernel = NUTS(matrix_model_edge_specific, target_accept_prob=0.7, max_tree_depth=10)
  mcmc_sampler = MCMC(kernel, num_warmup=300, num_samples=1000, num_chains=3,  progress_bar=True)

  # Run MCMC with edge-specific priors
  mcmc_sampler.run(
      jax.random.PRNGKey(42),
      obs_data=Rhat_df.values,
      pi0_ij=pi0_ij,
      pi_k_ij=pi_k_ij,
      pi_k=pi_slabs,       # Global mixture proportions for slab components
      sigma_grid=slab_scales,  # Slab variances
      U_lower=U_lower,
      V_lower=V_lower,
      D=D
  )

  inf_data = az.from_numpyro(mcmc_sampler)
  summary_df = az.summary(inf_data)
  summary_df.reset_index().to_csv("summary.csv", index=False)

  i_range = range(5)
  j_range = range(5)

  xbins = np.linspace(-1, 1, 200)
  fig = plt.figure(figsize=(20,20))#, layout="constrained")
  for i in i_range:
      for j in j_range:
          ax = fig.add_subplot(len(j_range), len(i_range), i*len(j_range) + j + 1)
          ax.hist(np.reshape(np.array(inf_data['posterior']['G'][:,:,i,j]), -1), histtype='step', density=True, bins=xbins, label='NUTS')
          #ax.legend(prop={'size': 6})
          ax.set_title(f'G[{i}, {j}]', fontsize=8)
          ax.tick_params(axis='both', which='major', labelsize=8)

  plt.savefig(path + 'my_plot_nuts.png') 
  #plt.show()

  posterior_samples = inf_data.posterior["G"].values
  chains, draws, G_dim_0, G_dim_1 = posterior_samples.shape
  flat_samples = posterior_samples.reshape(chains * draws, G_dim_0, G_dim_1)  # Shape: (5000, 10, 10)

  plt.figure(figsize=(5,4))
  posterior_mean = np.mean(flat_samples, axis=0)
  plt.imshow(posterior_mean, cmap="coolwarm")
  plt.colorbar()
  plt.title("Posterior Mean of G")
  plt.savefig(path + 'posterior_mean.png') 
  #plt.show()

  df1 = pd.DataFrame(posterior_mean)
  df1.to_csv(path + "posterior_mean.csv", header=False, index=False)

  # Posterior Inclusion Probability
  epsilon = 0.05  # Threshold for non-zero edge
  pip_matrix = np.mean(np.abs(flat_samples) > epsilon, axis=0)  # Shape: (G_dim_0, G_dim_1)
  plt.figure(figsize=(5,4))
  plt.imshow(pip_matrix, cmap="coolwarm")
  plt.colorbar()
  plt.title("Posterior Inclusion Probability (PIP)")
  plt.savefig(path + 'pip.png') 
  #plt.show()

  df2 = pd.DataFrame(pip_matrix)
  df2.to_csv(path + "pip_matrix.csv", header=False, index=False)


  # 3) Optional: Compute LFSR
  plt.figure(figsize=(5,4))
  lfsr_matrix = compute_lfsr(flat_samples)
  plt.imshow(lfsr_matrix, cmap="coolwarm", vmin=0, vmax=1)
  plt.colorbar()
  plt.title("Local False Sign Rate (LFSR)")
  plt.savefig(path + 'lfsr.png') 
  #plt.show()

  df3 = pd.DataFrame(lfsr_matrix)
  df3.to_csv(path + "lfsr_matrix.csv", header=False, index=False)
  # 4) If you also want to filter by LFSR (sign certainty),
  #    choose a small cutoff, e.g. lfsr <= 0.05:
  lfsr_cutoff = 0.2

  # Zero out entries where LFSR is too big:
  posterior_mean = np.mean(flat_samples, axis=0)  # shape (D, D)

  combined_matrix = posterior_mean
  combined_matrix[lfsr_matrix > lfsr_cutoff] = 0

  # Plot final
  plt.figure(figsize=(5,4))
  plt.imshow(combined_matrix, cmap="coolwarm")
  plt.colorbar()
  plt.title(f"Posterior Mean + LFSR <= {lfsr_cutoff}")
  plt.savefig(path + 'lfsr_0.2.png') 
  #plt.show()

  plt.figure(figsize=(5,4))
  plt.hist(lfsr_matrix.flatten(), bins=60, edgecolor="black", alpha=0.7)
  plt.xlabel("LFSR Values")
  plt.ylabel("Frequency")
  plt.title("Distribution of LFSR Values")
  plt.savefig(path + 'lfsr_dist.png') 
  #plt.show()

  lfsr_thresholds = np.concatenate([np.arange(0.00, 0.20, 0.02), np.arange(0.20, 1.05, 0.05)])
  for lfsr_cutoff in lfsr_thresholds:
      filtered_matrix = posterior_mean.copy()
      filtered_matrix[lfsr_matrix > lfsr_cutoff] = 0
      print(f"LFSR ≤ {lfsr_cutoff:.2f}: {np.count_nonzero(filtered_matrix)} nonzero entries")


  os.makedirs(path, exist_ok=True)

  lfsr_matrix = compute_lfsr(flat_samples)
  lfsr_thresholds = np.arange(0.00, 1.05, 0.05)  # Generate thresholds from 0.05 to 0.30

  # Save filtered matrices for different LFSR cutoffs
  output_files = []
  for lfsr_cutoff in lfsr_thresholds:
      combined_matrix = posterior_mean.copy()
      combined_matrix[lfsr_matrix > lfsr_cutoff] = 0  # Apply LFSR filtering

      filename = f"random1_{lfsr_cutoff:.2f}.csv"
      #filename = f"random_1_50_{lfsr_cutoff:.2f}.csv"
      filepath = os.path.join(path, filename) 
      
      pd.DataFrame(combined_matrix).to_csv(filepath, index=False, header=False, float_format="%.6f")
      output_files.append(filename)

  print("Generated LFSR-filtered CSV files:", output_files)




      
