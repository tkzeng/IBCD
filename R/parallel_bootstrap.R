library(inspre)
library(tidyverse)
library(hdf5r)
library(progress)
library(mc2d)
library(doParallel)
library(foreach)
library(progressr)
library(doRNG)

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
D <- 500
N_int <- 100
N_cont <- D*N_int
int_beta <- -2.7
graph <- "random"
v <- 0.3
p <- 0.08
DAG <- TRUE
C <- 0
num_simulations <- 1000
set.seed(42) 

data_dir <- "/Users/seongwoohan/Desktop/dontmessup/boostrap/sim_R/true_bootstrap/"

rdata_dir <- file.path(data_dir, "rdata")
xi_dir <- file.path(data_dir, "xi")
u_dir <- file.path(data_dir, "U")
v_dir <- file.path(data_dir, "V")
dirs <- list(data_dir, rdata_dir, xi_dir, u_dir, v_dir)
lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE)

dataset <- generate_dataset(D = D, N_cont = N_cont, N_int = N_int, int_beta = int_beta, graph = graph, v = v, p = p, DAG = DAG, C = C)
G <- dataset$G  # True direct effects matrix
R <- dataset$R  # True total causal effects matrix

#generate_data_inhibition <- inspre:::generate_data_inhibition
#dataset <- generate_data_inhibition(G, N_cont, N_int, int_beta, noise='gaussian')

Y_fixed <- dataset$Y
targets_fixed <- dataset$targets

# Save true matrices
write.csv(dataset$G, file = file.path(data_dir, "G_matrix.csv"), row.names = FALSE)
write.csv(dataset$R, file = file.path(data_dir, "R_matrix.csv"), row.names = FALSE)
write.csv(dataset$Y, file = file.path(data_dir, "Y_matrix.csv"), row.names = FALSE)

n_cores <- 8#80
cl      <- makeCluster(n_cores)
registerDoParallel(cl)

results <- foreach(sim = 1:num_simulations, .options.RNG = 42,
                   .packages = c("inspre", "mc2d", "tidyverse")) %dorng% {
                     #n <- nrow(Y_fixed)
                     #boot_idx <- sample(n, n, replace = TRUE)
                     #Y_boot <- Y_fixed[boot_idx, ]
                     #targets_boot <- targets_fixed[boot_idx]
                     
                     uniq_targets <- unique(targets_fixed)  # "control", "V1", … "V500"
                     boot_idx <- unlist(
                       lapply(uniq_targets, function(tg) {
                         id  <- which(targets_fixed == tg)     # row indices for this block
                         sample(id, length(id), replace = TRUE)# resample same size
                       }),
                       use.names = FALSE
                     )
                     
                     Y_boot       <- Y_fixed[boot_idx, ]           
                     targets_boot <- targets_fixed[boot_idx]  
                     
                     Y_C <- Y_boot[targets_boot == "control", ]
                     Sigma <- t(Y_C) %*% Y_C / nrow(Y_C)
                     xi <- Sigma^2
                     diag(xi) <- 0
                     
                     start_time <- Sys.time()
                     
                     genes <- paste0("V", 1:D)
                     U_diag <- numeric(D)
                     V_list <- vector("list", D)
                     R_hat <- matrix(0, nrow = D, ncol = D)
                     
                     for (i in seq_along(genes)) {
                       gene <- genes[i]
                       res <- multiple_iv_reg_UV(gene, Y_boot, targets_boot)
                       U_diag[i] <- res$U_i
                       V_list[[i]] <- res$V_hat
                       R_hat[i, ] <- unlist(res$beta_se[1, grep("_beta_hat$", names(res$beta_se))])
                       
                       # Optional: log every 50 genes
                       if (i %% 5 == 0 || i == D) {
                         cat(sprintf("Sim %03d – Gene %s (%d/%d)\n", sim, gene, i, D),
                             file = file.path(data_dir, "bootstrap_progress.log"), append = TRUE)
                       }
                     }
                     
                     end_time <- Sys.time()
                     duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
                     
                     cat(sprintf("Finished bootstrap %03d in %.2f sec\n", sim, duration),
                         file = file.path(data_dir, "bootstrap_progress.log"), append = TRUE)
                     
                     U <- diag(U_diag, D)
                     V <- Reduce("+", V_list) / D
                     
                     # Save to disk
                     write.csv(U, file = file.path(u_dir, sprintf("U_%03d.csv", sim)), row.names = FALSE)
                     write.csv(V, file = file.path(v_dir, sprintf("V_%03d.csv", sim)), row.names = FALSE)
                     write.csv(R_hat, file = file.path(rdata_dir, sprintf("R_hat_%03d.csv", sim)), row.names = FALSE)
                     write.csv(xi, file = file.path(xi_dir, sprintf("xi_%03d.csv", sim)), row.names = FALSE)
                     
                     return(list(R_hat = R_hat, U = U, V = V, xi = xi))
                   }

stopCluster(cl)

# Aggregate
xi_mean <- Reduce("+", lapply(results, `[[`, "xi")) / num_simulations
xi_norm <- xi_mean / sqrt(sum(xi_mean^2))
write.csv(xi_norm, file = file.path(data_dir, "xi_500.csv"))

R_mean <- Reduce("+", lapply(results, `[[`, "R_hat")) / num_simulations
write.csv(R_mean, file = file.path(data_dir, "R_500.csv"), row.names = FALSE)

U_mean <- Reduce("+", lapply(results, `[[`, "U")) / num_simulations
write.csv(U_mean, file = file.path(data_dir, "U_500.csv"), row.names = FALSE)

V_mean <- Reduce("+", lapply(results, `[[`, "V")) / num_simulations
write.csv(V_mean, file = file.path(data_dir, "V_500.csv"), row.names = FALSE)
