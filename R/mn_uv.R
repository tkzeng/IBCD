library(inspre)
library(tidyverse)
library(hdf5r)
library(progress)
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

# 1
# No cycles, no confounding, random
# High density, large effects, strong instruments
D <- 10
N_int <- 100
N_cont <- D*N_int
int_beta <- -2.7
graph <- "random"
v <- 0.3
p <- 0.08
DAG <- TRUE
C <- 0
num_simulations <- 100
set.seed(42) 

data_dir <- "/home/seong/lr/"
rdata_dir <- file.path(data_dir, "rdata")
xi_dir <- file.path(data_dir, "xi")
u_dir <- file.path(data_dir, "U")       
v_dir <- file.path(data_dir, "V")       
log_file <- file.path(data_dir, "sim_log.txt")

dir.create(data_dir , recursive = TRUE, showWarnings = FALSE)
dir.create(rdata_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(xi_dir   , recursive = TRUE, showWarnings = FALSE)
dir.create(u_dir    , recursive = TRUE, showWarnings = FALSE)
dir.create(v_dir    , recursive = TRUE, showWarnings = FALSE)
if (file.exists(log_file)) file.remove(log_file)

dataset <- generate_dataset(D = D, N_cont = N_cont, N_int = N_int, int_beta = int_beta, graph = graph, v = v, p = p, DAG = DAG, C = C)
G <- dataset$G  # True direct effects matrix
R <- dataset$R  # True total causal effects matrix

write.csv(G, file = file.path(data_dir, "G_matrix.csv"), row.names = FALSE)
write.csv(R, file = file.path(data_dir, "R_matrix.csv"), row.names = FALSE)

R_hat_list <- vector("list", num_simulations)
xi_list <- vector("list", num_simulations)

for (sim in 1:num_simulations) {
  generate_data_inhibition <- inspre:::generate_data_inhibition
  dataset <- generate_data_inhibition(G, N_cont, N_int, int_beta, noise='gaussian')
  
  Y       <- dataset$Y
  targets <- dataset$targets
  
  Y_C   <- Y[targets == "control", ]          # controls only
  Sigma <- t(Y_C) %*% Y_C / nrow(Y_C)
  xi    <- Sigma^2                            # element‑wise square
  diag(xi) <- 0                               # zero the diagonal
  xi_list[[sim]] <- xi
  
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
    res_list <- multiple_iv_reg_UV(genes[i], Y, targets)
    U_diag[i]        <- res_list$U_i
    V_list[[i]]      <- res_list$V_hat
    beta_se_tbl[[i]] <- res_list$beta_se
    beta_hat_vec <- unlist(res_list$beta_se[1, grep("_beta_hat$", names(res_list$beta_se))])
    R_hat[i, ] <- as.numeric(beta_hat_vec)
    pb$tick()
  }
  
  U <- diag(U_diag, D, D)
  V <- Reduce("+", V_list) / D
  
  write.csv(U, file = file.path(u_dir, paste0("U_", sim, ".csv")), row.names = FALSE)
  write.csv(V, file = file.path(v_dir, paste0("V_", sim, ".csv")), row.names = FALSE)
  
  # Store the R_hat matrix
  R_hat_list[[sim]] <- R_hat
  write.csv(R_hat, file = file.path(rdata_dir, paste0("R_hat_", sim, ".csv")), row.names = FALSE)
  write.csv(xi, file = file.path(xi_dir, paste0("xi_", sim, ".csv")), row.names = FALSE)
}

# Average results
xi_mean <- Reduce("+", xi_list) / num_simulations
xi_mean_normalized <- xi_mean / sqrt(sum(xi_mean^2))
write.csv(xi_mean_normalized, file = file.path(data_dir, "xi_500.csv"))

# Compute the representative R_hat (average over simulations)
R_hat_representative <- Reduce("+", R_hat_list) / num_simulations
write.csv(R_hat_representative, file = file.path(data_dir, "R_500.csv"), row.names = FALSE)
