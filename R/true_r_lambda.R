D <- 500
N_int <- 100
N_cont <- D*N_int
int_beta <- -2.7
graph <- "random"
v <- 0.3
p <- 0.08
DAG <- TRUE
C <- 0
set.seed(42) 

data_dir <- "/Users/seongwoohan/Desktop/dontmessup/boostrap/sim_R/true_uvrg/"
dataset <- generate_dataset(D = D, N_cont = N_cont, N_int = N_int, int_beta = int_beta, graph = graph, v = v, p = p, DAG = DAG, C = C)
G <- dataset$G  # True direct effects matrix
R <- dataset$R  # True total causal effects matrix

write.csv(dataset$G, file = file.path(data_dir, "G_matrix.csv"), row.names = FALSE)
write.csv(dataset$R, file = file.path(data_dir, "R_matrix.csv"), row.names = FALSE)

Y_fixed <- dataset$Y
targets_fixed <- dataset$targets

genes   <- paste0("V", 1:D)
U_diag <- numeric(D)               # store diagonal entries
V_list <- vector("list", D)        # store V_hat per gene
beta_se_tbl <- vector("list", D)   # store beta_se data frames
R_hat <- matrix(0, nrow = D, ncol = D)

pb <- progress_bar$new(
  format = "[:bar] :current/:total (:percent) | Elapsed: :elapsed | ETA: :eta",
  total = length(genes), clear = FALSE, width = 80
)

for (i in seq_along(genes)) {
  message(sprintf("Sim %d / %d", i, D))
  res_list <- multiple_iv_reg_UV(genes[i], Y_fixed, targets_fixed)
  U_diag[i]        <- res_list$U_i
  V_list[[i]]      <- res_list$V_hat
  beta_se_tbl[[i]] <- res_list$beta_se
  beta_hat_vec <- unlist(res_list$beta_se[1, grep("_beta_hat$", names(res_list$beta_se))])
  R_hat[i, ] <- as.numeric(beta_hat_vec)
  pb$tick()
}

U <- diag(U_diag, D, D)
V <- Reduce("+", V_list) / D

write.csv(U, file = file.path(data_dir, paste0("U_true.csv")), row.names = FALSE)
write.csv(V, file = file.path(data_dir, paste0("V_true.csv")), row.names = FALSE)
write.csv(R_hat, file = file.path(data_dir, "R_true.csv"), row.names = FALSE)


n  <- N_cont + N_int * D          # total rows that produced R0
r       <- R_hat[upper.tri(R_hat)]                  # off-diagonals
var_r   <- ((1 - r^2)^2) / (n - 1)            # Var(r_ij)

num <- sum(var_r)                             # Σ Var(r_ij)
den <- sum(r^2)                               # Σ r_ij^2
lam <- max(0, min(1, num / den))              # λ̂★

cat(sprintf("λ̂★ = %.6f\n", lam))
