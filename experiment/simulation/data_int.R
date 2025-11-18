library(inspre)
library(tidyverse)
library(hdf5r)
library(progress)
library(mc2d)
library(doParallel)
library(foreach)
library(progressr)
library(doRNG)
library(tidyverse)
library(mc2d)

multiple_iv_reg_UV <- function(target, .X, .targets){
  inst_obs <- .targets == target
  control_obs <- .targets == "control"
  n_target <- sum(inst_obs)
  n_control <- sum(control_obs)
  n_total <- n_target + n_control
  X_target <- .X[inst_obs, ]
  X_control <- .X[control_obs, ]
  this_feature <- which(colnames(.X) == target)
  X_exp_target <- X_target[, this_feature]
  X_exp_control <- X_control[, this_feature]
  beta_inst_obs <- sum(X_exp_target)/n_target
  sums_target <- colSums(X_target)
  beta_hat <- sums_target/sums_target[this_feature]
  resid <- rbind(X_target - outer(X_exp_target, beta_hat), X_control - outer(X_exp_control, beta_hat))
  V_hat <- (t(resid)%*%resid)/nrow(resid)
  se_hat <- sqrt(diag(V_hat)/(n_target*beta_inst_obs**2))
  names(beta_hat) <- paste(names(beta_hat), "beta_hat", sep="_")
  names(se_hat) <- paste(names(se_hat), "se_hat", sep="_")
  return(list(
    beta_se = as.data.frame(c(list(target = target, beta_obs = beta_inst_obs), 
                              as.list(c(beta_hat, se_hat)))),
    U_i=1/(n_target*beta_inst_obs**2),
    V_hat=V_hat
  ))
  # return(beta_se=as.data.frame(c(list(target = target, beta_obs = beta_inst_obs), as.list(c(beta_hat, se_hat)))), U_i=1/n_target*beta_inst_obs**2, V_hat=V_hat)
}

base_dir <- file.path(getwd(), "int")

base_dir <-  "/Users/seongwoohan/Desktop/r_py"
int_beta_default <- -2 #-2 for normal, -1.5 for ablation 
v_default <- 0.25 #0.25 for normal, 0.2 for ablation 
graph_type <- "scalefree" # "scalefree" #"random"
DAG_flag <- TRUE
C_default <- 0
replicate_seeds <- 42:51 

run_one <- function(D, N_int, p, seed, out_dir,
                    int_beta = int_beta_default,
                    v = v_default,
                    graph = graph_type,
                    DAG = DAG_flag,
                    C = C_default) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf("[D=%d | N_int=%d | p=%.3f | seed=%d] -> %s",
                  D, N_int, p, seed, out_dir))
  set.seed(seed)
  
  N_cont <- D * N_int
  
  # --- Generate dataset ---
  dataset <- generate_dataset(
    D = D,
    N_cont = N_cont,
    N_int = N_int,
    int_beta = int_beta,
    graph = graph,
    v = v,
    p = p,
    DAG = DAG,
    C = C
  )
  
  G <- dataset$G
  R <- dataset$R
  Y_fixed <- dataset$Y
  targets_fixed <- dataset$targets
  writeLines(targets_fixed, file.path(out_dir, "targets.txt"))
  
  Y_C <- Y_fixed[targets_fixed == "control", ]
  Sigma <- t(Y_C) %*% Y_C / nrow(Y_C)
  xi <- Sigma^2
  diag(xi) <- 0
  xi_norm <- xi / sqrt(sum(xi^2))
  
  write.csv(G, file = file.path(out_dir, "G_matrix.csv"), row.names = FALSE)
  write.csv(R, file = file.path(out_dir, "R_matrix.csv"), row.names = FALSE)
  write.csv(Y_fixed, file = file.path(out_dir, "Y_matrix.csv"), row.names = FALSE)
  write.csv(xi_norm, file = file.path(out_dir, "xi_true.csv"), row.names = FALSE)
  
  genes <- paste0("V", 1L:D)
  U_diag <- numeric(D)
  V_sum <- matrix(0, D, D)
  R_hat <- matrix(0, D, D)
  SE_hat <- matrix(0, D, D)
  
  for (i in seq_along(genes)) {
    gene <- genes[i]
    message("Running IV for target: ", gene)
    res <- multiple_iv_reg_UV(gene, Y_fixed, targets_fixed)
    U_diag[i] <- res$U_i
    V_sum <- V_sum + res$V_hat
    R_hat[i, ] <- unlist(res$beta_se[1, grep("_beta_hat$", names(res$beta_se))])
    SE_hat[i, ] <- unlist(res$beta_se[1, grep("_se_hat$", names(res$beta_se))])
  }
  
  U <- diag(U_diag)
  V <- V_sum / D
  write.csv(R_hat, file = file.path(out_dir, "R_true.csv"), row.names = FALSE)
  write.csv(SE_hat, file = file.path(out_dir, "SE_hat.csv"), row.names = FALSE)
  write.csv(U,      file = file.path(out_dir, "U_true.csv"),    row.names = FALSE)
  write.csv(V,      file = file.path(out_dir, "V_true.csv"),    row.names = FALSE)
}

out_dir <- file.path(base_dir)
run_one(D = 50, N_int = 100, p = 0.10, seed = 42, out_dir = out_dir)



D_50 <- 50
p_50 <- 0.10
subfolders_50d <- c(5, 15, 25, 50, 75, 100)  

for (Nint in subfolders_50d) {
  for (seed in replicate_seeds) {
    out_dir <- file.path(base_dir, "50d", as.character(Nint), "random", as.character(seed))
    run_one(D = D_50, N_int = Nint, p = p_50, seed = seed, out_dir = out_dir)
  }
}


D_150 <- 150
Nint_150 <- 100
p_150 <- 0.033

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "150d", "random", as.character(seed))
  run_one(D = D_150, N_int = Nint_150, p = p_150, seed = seed, out_dir = out_dir)
}


D_250 <- 250
Nint_250 <- 100
p_250 <- 0.02

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "250d", "random", as.character(seed))
  run_one(D = D_250, N_int = Nint_250, p = p_250, seed = seed, out_dir = out_dir)
}


D_500 <- 500
Nint_500 <- 100
p_500 <- 0.01

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "500d", "random", as.character(seed))
  run_one(D = D_500, N_int = Nint_500, p = p_500, seed = seed, out_dir = out_dir)
}

message("All datasets generated and saved.")

#######################  scale free ########################## 
######## ######## ######## ######## ######## ######## ######## 

D_50 <- 50
p_50 <- 0.066
subfolders_50d <- c(5, 15, 25, 50, 75, 100)  # N_int values & folder names

for (Nint in subfolders_50d) {
  for (seed in replicate_seeds) {
    out_dir <- file.path(base_dir, "50d", as.character(Nint), "sf", as.character(seed))
    run_one(D = D_50, N_int = Nint, p = p_50, seed = seed, out_dir = out_dir)
  }
}

# -----------------------------
D_150 <- 150
Nint_150 <- 100
p_150 <- 0.108

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "150d", "sf", as.character(seed))
  run_one(D = D_150, N_int = Nint_150, p = p_150, seed = seed, out_dir = out_dir)
}

# -----------------------------

D_250 <- 250
Nint_250 <- 100
p_250 <- 0.124

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "250d", "sf", as.character(seed))
  run_one(D = D_250, N_int = Nint_250, p = p_250, seed = seed, out_dir = out_dir)
}

# -----------------------------

D_500 <- 500
Nint_500 <- 100
p_500 <- 0.139

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "500d", "sf", as.character(seed))
  run_one(D = D_500, N_int = Nint_500, p = p_500, seed = seed, out_dir = out_dir)
}

message("All datasets generated and saved.")
