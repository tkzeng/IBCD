library(inspre)
library(tidyverse)
library(hdf5r)
library(mc2d)
library(foreach)
library(doRNG)
library(tidyverse)

base_dir <- file.path(getwd(), "ablation_d")
int_beta <- -1.5
v        <- 0.2
dims <- list(
  `150d` = list(D = 150, p = 0.033),
  `250d` = list(D = 250, p = 0.020),
  `500d` = list(D = 500, p = 0.010)
)

generate_and_save_random <- function(dim_tag, seed_num, D, int_beta, v, p, base_dir) {
  set.seed(seed_num)
  
  dim_dir  <- file.path(base_dir, dim_tag)    
  seed_dir <- file.path(dim_dir, as.character(seed_num))
  lapply(list(dim_dir, seed_dir), dir.create, recursive = TRUE, showWarnings = FALSE)
  
  N_int  <- 100
  N_cont <- D * N_int
  graph  <- "random"
  DAG    <- TRUE
  C      <- 0
  
  dataset <- generate_dataset(
    D = D, N_cont = N_cont, N_int = N_int,
    int_beta = int_beta, graph = graph, v = v, p = p, DAG = DAG, C = C
  )
  G <- dataset$G
  R <- dataset$R
  Y_fixed       <- dataset$Y
  targets_fixed <- dataset$targets
  
  writeLines(targets_fixed, file.path(seed_dir, "targets.txt"))
  
  Y_C   <- Y_fixed[targets_fixed == "control", ]
  Sigma <- t(Y_C) %*% Y_C / nrow(Y_C)
  xi    <- Sigma^2
  diag(xi) <- 0
  xi_norm <- xi / sqrt(sum(xi^2))
  
  write.csv(G,        file = file.path(seed_dir, "G_matrix.csv"), row.names = FALSE)
  write.csv(R,        file = file.path(seed_dir, "R_matrix.csv"), row.names = FALSE)
  write.csv(Y_fixed,  file = file.path(seed_dir, "Y_matrix.csv"), row.names = FALSE)
  write.csv(xi_norm,  file = file.path(seed_dir, "xi_true.csv"),  row.names = FALSE)
  
  genes  <- paste0("V", seq_len(D))
  U_diag <- numeric(D)
  V_sum  <- matrix(0, D, D)
  R_hat  <- matrix(0, D, D)
  SE_hat <- matrix(0, D, D)
  
  proj_y_list  <- vector("list", D)
  residual_list <- vector("list", D)
  
  for (i in seq_along(genes)) {
    gene <- genes[i]
    message(sprintf("[%s | seed %d] IV for %s (%d/%d)", dim_tag, seed_num, gene, i, D))
    res <- multiple_iv_reg_UV(gene, Y_fixed, targets_fixed)
    
    U_diag[i] <- res$U_i
    V_sum     <- V_sum + res$V_hat
    R_hat[i, ]  <- unlist(res$beta_se[1, grep("_beta_hat$", names(res$beta_se))])
    SE_hat[i, ] <- unlist(res$beta_se[1, grep("_se_hat$",   names(res$beta_se))])
    proj_y_list[[i]] <- res$proj_y
    residual_list[[i]] <- res$resid
  }
  
  U <- diag(U_diag)
  V <- V_sum / D
  
  write.csv(R_hat, file = file.path(seed_dir, "R_true.csv"),   row.names = FALSE)
  write.csv(SE_hat, file = file.path(seed_dir, "SE_hat.csv"),  row.names = FALSE)
  write.csv(U,      file = file.path(seed_dir, "U_true.csv"),  row.names = FALSE)
  write.csv(V,      file = file.path(seed_dir, "V_true.csv"),  row.names = FALSE)
  }

for (tag in names(dims)) {
  D <- dims[[tag]]$D
  p <- dims[[tag]]$p
  for (seed in 42:51) {
    message(sprintf("\n--- %s | seed %d | D=%d | int_beta=%.1f | v=%.1f | p=%.3f ---",
                    tag, seed, D, int_beta, v, p))
    generate_and_save_random(tag, seed, D = D, int_beta = int_beta, v = v, p = p,
                             base_dir = base_dir)
    gc()
  }
}

