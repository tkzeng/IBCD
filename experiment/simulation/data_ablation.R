library(inspre)
library(hdf5r)
library(mc2d)
library(foreach)
library(doRNG)

compute_S_hat <- function(proj_y_list, residual_list) {
  D <- length(proj_y_list)
  n <- nrow(proj_y_list[[1]])
  
  # Stack proj_y (n x D) and residuals (n x D x D)
  proj_mat <- do.call(cbind, lapply(proj_y_list, function(x) x[, 1]))  # (n, D)
  res_array <- array(unlist(residual_list), dim = c(n, D, D))          # (n, D, D)
  
  # Compute proj cov and var
  proj_cov <- cov(proj_mat)                         # (D, D)
  proj_var <- apply(proj_mat, 2, var)               # (D,)
  
  # Flatten residuals: (n, D^2)
  res_flat <- matrix(aperm(res_array, c(1, 2, 3)), nrow = n)  # (n, D^2)
  res_cov <- cov(res_flat)                                   # (D^2, D^2)
  
  # Preallocate S_hat
  S_hat <- matrix(0, nrow = D * D, ncol = D * D)
  
  # Compute S_hat[a,b] = proj_cov[i,k] * res_cov[j,l] / (n * var_i * var_k)
  for (a in 1:(D * D)) {
    i <- ((a - 1) %/% D) + 1
    j <- ((a - 1) %% D) + 1
    
    if (j == 1 && i %% 10 == 0) {   # every 10 rows, not every row
      message(sprintf("Computing row %d of %d (i=%d)", a, D*D, i))
    }
    
    for (b in 1:(D * D)) {
      k <- ((b - 1) %/% D) + 1
      l <- ((b - 1) %% D) + 1
      denom <- proj_var[i] * proj_var[k] * n
      if (denom > 0) {
        S_hat[a, b] <- proj_cov[i, k] * res_cov[(i - 1) * D + j, (k - 1) * D + l] / denom
      }
    }
  }
  
  # Add names
  index_pairs <- expand.grid(i = 1:D, j = 1:D)
  rownames(S_hat) <- apply(index_pairs, 1, function(x) paste0("V", x[1], "_", x[2]))
  colnames(S_hat) <- rownames(S_hat)
  
  return(S_hat)
}



settings <- list(
  list(comment = "#random", int_beta = -1.5, v = 0.2, p = 0.1)
)

#settings <- list(
#  list(comment = "#scalefree", int_beta = -1.5, v = 0.2, p = 0.066)
#)


generate_and_save_random <- function(setting_idx, seed_num, int_beta, v, p) {
  set.seed(seed_num)
  base_dir <- file.path(getwd(), "ablation")
  #setting_dir <- file.path(base_dir, paste0("sf", setting_idx))      
  setting_dir <- file.path(base_dir, paste0("r", setting_idx))      
  seed_dir <- file.path(setting_dir, paste0(seed_num))  
  dirs <- list(setting_dir, seed_dir)
  
  lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE)
  
  D <- 50
  N_int <- 100
  N_cont <- D * N_int
  graph <- "random" # random scalefree
  DAG <- TRUE
  C <- 0
  
  dataset <- generate_dataset(D = D, N_cont = N_cont, N_int = N_int, int_beta = int_beta, graph = graph, v = v, p = p, DAG = DAG, C = C)
  G <- dataset$G
  R <- dataset$R
  Y_fixed <- dataset$Y
  targets_fixed <- dataset$targets
  
  writeLines(targets_fixed, file.path(seed_dir, "targets.txt"))
  
  Y_C <- Y_fixed[targets_fixed == "control", ]
  Sigma <- t(Y_C) %*% Y_C / nrow(Y_C)
  xi <- Sigma^2
  diag(xi) <- 0
  xi_norm <- xi / sqrt(sum(xi^2))
  
  write.csv(G, file = file.path(seed_dir, "G_matrix.csv"), row.names = FALSE)
  write.csv(R, file = file.path(seed_dir, "R_matrix.csv"), row.names = FALSE)
  write.csv(Y_fixed, file = file.path(seed_dir, "Y_matrix.csv"), row.names = FALSE)
  write.csv(xi_norm, file = file.path(seed_dir, "xi_true.csv"), row.names = FALSE)
  
  genes <- paste0("V", 1L:D)
  U_diag <- numeric(D)
  V_sum <- matrix(0, D, D)
  R_hat <- matrix(0, D, D)
  SE_hat <- matrix(0, D, D)
  #####
  proj_y_list <- list()
  residual_list <- list()
  #####
  for (i in seq_along(genes)) {
    gene <- genes[i]
    message("Running IV for target: ", gene)
    res <- multiple_iv_reg_UV(gene, Y_fixed, targets_fixed)
    U_diag[i] <- res$U_i
    V_sum <- V_sum + res$V_hat
    R_hat[i, ] <- unlist(res$beta_se[1, grep("_beta_hat$", names(res$beta_se))])
    SE_hat[i, ] <- unlist(res$beta_se[1, grep("_se_hat$", names(res$beta_se))])
    #####
    proj_y_list[[i]] <- res$proj_y
    residual_list[[i]] <- res$resid
    #####
  }
  
  U <- diag(U_diag)
  V <- V_sum / D
  write.csv(R_hat, file = file.path(seed_dir, "R_true.csv"), row.names = FALSE)
  write.csv(SE_hat, file = file.path(seed_dir, "SE_hat.csv"), row.names = FALSE)
  
  write.csv(U,      file = file.path(seed_dir, "U_true.csv"),    row.names = FALSE)
  write.csv(V,      file = file.path(seed_dir, "V_true.csv"),    row.names = FALSE)
  S_hat <- compute_S_hat(proj_y_list, residual_list)
  write.csv(S_hat, file = file.path(seed_dir, "S_true.csv"))
  
}

for (setting_idx in 1:length(settings)) {
  setting <- settings[[setting_idx]]
  cat("\n\n", setting$comment, "\n\n")  # print the setting comment
  for (seed in 42:51) {
    message("Processing setting ", setting_idx, ", seed ", seed)
    generate_and_save_random(setting_idx, seed, int_beta = setting$int_beta, v = setting$v, p = setting$p)
    gc()
  }
}

