source("inspre.R")
source("simulate.R")

# simulate_rand_no_cycles_no_confound.Rmd
# No cycles, no confounding, random
# High density, large effects, strong instruments

# Parameters
D <- 10  # Number of noades 
N_int <- 100  # Intervention samples per node
N_cont <- D * N_int  # Number of control samples
int_beta <- -2.7  # Intervention effect size
graph <- "random"  # Graph type
v <- 0.3  # PERT mode
p <- 0.08  # Edge probability
DAG <- TRUE  # Directed Acyclic Graph?
C <- 0  # Confounders

num_simulations <- 100  # Number of simulations for estimating S^
set.seed(42)

# Step 1: Generate a single G and R (true network and causal effects)
dataset <- generate_dataset(D = D, N_cont = N_cont, N_int = N_int, int_beta = int_beta, graph = graph, v = v, p = p, DAG = DAG, C = C)
G <- dataset$G  # True direct effects matrix
R <- dataset$R  # True total causal effects matrix

# Step 2: Simulate multiple datasets and estimate R^
R_hat_list <- vector("list", num_simulations)

for (sim in 1:num_simulations) {
  # Generate a new dataset based on the same G
  dataset <- generate_dataset(D = D, N_cont = N_cont, N_int = N_int, int_beta = int_beta,
                              graph = graph, v = v, p = p, DAG = DAG, C = C)
  Y <- dataset$Y
  targets <- dataset$targets
  
  # Estimate R^ using instrumental variables
  R_hat <- matrix(0, nrow = D, ncol = D)
  for (i in 1:D) {
    target <- paste0("V", i)
    res <- multiple_iv_reg(target, Y, targets)
    R_hat[i, ] <- as.numeric(res[grep("beta_hat", names(res))])
  }
  
  # Store the R_hat matrix
  R_hat_list[[sim]] <- R_hat
  
  # Save the R_hat matrix to a file
  # write.csv(R_hat, paste0("R_hat_sim_", sim, ".csv"), row.names = FALSE)
}

# Compute the representative R_hat (average over simulations)
R_hat_representative <- Reduce("+", R_hat_list) / num_simulations

# Step 3: Compute the empirical covariance matrix S^
calculate_covariance_matrix_S <- function(R_hat_list, D) {
  # Initialize the S^ matrix with dimensions (DxD)x(DxD)
  S_hat <- array(0, dim = c(D, D, D, D))
  
  # Loop over all pairs (i,j) and (k,l)
  for (i in 1:D) {
    for (j in 1:D) {
      for (k in 1:D) {
        for (l in 1:D) {
          # Extract distributions of R^_ij and R^_kl from all simulations
          R_ij_values <- sapply(R_hat_list, function(R_hat) R_hat[i, j])
          R_kl_values <- sapply(R_hat_list, function(R_hat) R_hat[k, l])
          
          S_hat[i, j, k, l] <- cov(R_ij_values, R_kl_values)
        }
      }
    }
  }
  
  return(S_hat)
}

# Calculate the covariance matrix S^ including self-effects
S_hat <- calculate_covariance_matrix_S(R_hat_list, D)

# Step 4: Reshape S_hat into a 2D 9x9 matrix
# Create all (i,j) pairs
pairs <- expand.grid(i = 1:D, j = 1:D)
num_pairs <- nrow(pairs)  # Should be D*D = 9 for D=3

# Initialize the reshaped S_hat matrix
S_hat_reshaped <- matrix(0, nrow = num_pairs, ncol = num_pairs)

# Assign meaningful row and column names
pair_names <- apply(pairs, 1, function(x) paste0("V", x[1], "_", x[2]))
rownames(S_hat_reshaped) <- pair_names
colnames(S_hat_reshaped) <- pair_names

# Fill the reshaped S_hat matrix
for (row_idx in 1:num_pairs) {
  for (col_idx in 1:num_pairs) {
    i <- pairs$i[row_idx]
    j <- pairs$j[row_idx]
    k <- pairs$i[col_idx]
    l <- pairs$j[col_idx]
    S_hat_reshaped[row_idx, col_idx] <- S_hat[i, j, k, l]
  }
}

# Step 5: Write CSV files
write.csv(G, "G_matrix.csv", row.names = FALSE)
write.csv(R, "R_matrix.csv", row.names = FALSE)
write.csv(R_hat_representative, "R_hat_matrix.csv", row.names = FALSE)
write.csv(S_hat_reshaped, "S_hat_matrix.csv", row.names = TRUE)  
