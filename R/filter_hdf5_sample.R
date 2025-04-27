library(inspre)
library(tidyverse)
library(hdf5r)
library(stringr)
library(doParallel)
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

data_dir <- "/Users/seongwoohan/Desktop/dontmessup/boostrap/sim_R/true_bootstrap/gwps/onemoretime/"
rdata_dir <- file.path(data_dir, "rdata")
xi_dir <- file.path(data_dir, "xi")
u_dir <- file.path(data_dir, "U")       
v_dir <- file.path(data_dir, "V")       

dir.create(data_dir , recursive = TRUE, showWarnings = FALSE)
dir.create(rdata_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(xi_dir   , recursive = TRUE, showWarnings = FALSE)
dir.create(u_dir    , recursive = TRUE, showWarnings = FALSE)
dir.create(v_dir    , recursive = TRUE, showWarnings = FALSE)

# 1. Load your gene list from CSV
k562_gwps_norm_sc_fn <- "/Users/seongwoohan/Desktop/inspre_bayes/R/K562_gwps_normalized_singlecell_01.h5ad"
#k562_essential_norm_sc_fn <- "/Users/seongwoohan/Desktop/inspre_bayes/R/K562_essential_normalized_singlecell_01.h5ad"

#k562_gwps_norm_sc_fn <- "/data/long_read/R/35774440"
#k562_essential_norm_sc_fn <-"/data/long_read/R/35773075"

gene_list <- read_csv("/Users/seongwoohan/Desktop/inspre_bayes/R/test.csv", col_names = TRUE)
gene_list <- gene_list$common_genes

# 2. Extract ENSG IDs from the gene list (last part of each string)
ensg_ids <- str_extract(gene_list, "ENSG[0-9]{11}")

# 3. Load HDF5 data
process_start <- Sys.time()
#hfile_sc <- H5File$new(k562_essential_norm_sc_fn, "r")
hfile_sc <- H5File$new(k562_gwps_norm_sc_fn, "r")
obs <- parse_hdf5_df(hfile_sc, "obs")
var <- parse_hdf5_df(hfile_sc, "var")

# 4. Filter var for the ENSG IDs in your gene list
genes_to_keep <- var %>% filter(gene_id %in% ensg_ids)
rows_to_keep <- match(genes_to_keep$gene_id, var$gene_id)

# 5. Find cells (columns of X) to keep:
# - Either non-targeting OR gene_id in your list
cells_to_keep <- obs %>%
  filter(gene == "non-targeting" | gene_id %in% ensg_ids)
cols_to_keep <- which(obs$gene == "non-targeting" | obs$gene_id %in% ensg_ids)

# 6. Read only the subset of X (genes × cells)
#    Note: X is [genes, cells] = [rows, cols]
X_sub <- hfile_sc[["X"]][rows_to_keep, cols_to_keep]

# 7. Annotate rows and columns for downstream use
rownames(X_sub) <- var$gene_name[rows_to_keep]  # or gene_id
colnames(X_sub) <- obs$gene_transcript[cols_to_keep]

# Step 1: Transpose X_sub so rows = cells, columns = genes
X <- t(X_sub)  # Now rows are cells, columns are genes

# Step 2: Construct the vector of targets per cell
# Extract gene name from column name like "713_AURKAIP1_P1P2_ENSG00000175756"
targets_raw <- colnames(X_sub)
targets_clean <- ifelse(
  grepl("non-targeting", targets_raw),
  "control",
  sapply(strsplit(targets_raw, "_"), `[`, 2)
)


X_control <- X[targets_clean == "control", , drop = FALSE]
# Compute coexpression matrix from control cells
Sigma <- t(X_control) %*% X_control / nrow(X_control)
# Elementwise square and zero out the diagonal
xi <- Sigma^2
diag(xi) <- 0

write.csv(xi, file = file.path(xi_dir, "xi_true.csv"), row.names = FALSE)

D <- ncol(X) 
#genes   <- paste0("V", 1L:D) 
U_diag  <- numeric(D)
V_sum   <- matrix(0, D, D)         # running total of V̂
R_hat   <- matrix(0, D, D)

for (i in seq_len(D)) {
  target <- colnames(X)[i]
  message("Running IV for target: ", target)
  res <- multiple_iv_reg_UV(target, X, targets_clean)  # <<<< HERE: use X and targets_clean
  U_diag[i] <- res$U_i
  V_sum     <- V_sum + res$V_hat
  R_hat[i, ] <- unlist(res$beta_se[1, grep("_beta_hat$", names(res$beta_se))])
}

U <- diag(U_diag)
V <- V_sum / D

write.csv(R_hat, file = file.path(rdata_dir, "R_true.csv"), row.names = FALSE)
write.csv(U,      file = file.path(u_dir,     "U_true.csv"),     row.names = FALSE)
write.csv(V,      file = file.path(v_dir,     "V_true.csv"),     row.names = FALSE)

######################
# parallel computing #
#######################
core <- 8
cl <- makeCluster(core)
registerDoParallel(cl)
registerDoRNG(seed = 42)

num_simulations <- 1000


results <- foreach(sim = 1:num_simulations, .options.RNG = 42,
                   .packages = c("inspre", "mc2d", "tidyverse")) %dorng% {
                     
                     start_time <- Sys.time()
                     
                     ## 1) stratified bootstrap
                     boot_idx <- unlist(lapply(unique(targets_clean), function(tg) {
                       id <- which(targets_clean == tg)
                       sample(id, length(id), replace = TRUE)
                     }), use.names = FALSE)
                     
                     X_bs <- X[boot_idx, , drop = FALSE]
                     t_bs <- targets_clean[boot_idx]
                     
                     ## 2) xi from controls
                     ctrl_bs  <- X_bs[t_bs == "control", , drop = FALSE]
                     Sigma_bs <- t(ctrl_bs) %*% ctrl_bs / nrow(ctrl_bs)
                     xi_bs    <- Sigma_bs^2
                     diag(xi_bs) <- 0
                     
                     genes      <- colnames(X_bs)  
                     D_genes    <- length(genes)
                     R_hat      <- matrix(0, nrow = D_genes, ncol = D_genes)
                     
                     for (i in seq_len(D_genes)) {
                       target_gene <- genes[i]  
                       message("Sim ", sim, " – Running IV for target: ", target_gene)
                       multiple_iv_reg <- inspre:::multiple_iv_reg
                       res <- multiple_iv_reg(target_gene, X_bs, t_bs)
                       
                       beta_idx      <- grep("_beta_hat$", names(res))  
                       R_hat[i, ]    <- as.numeric(res[beta_idx])
                       
                       if (i %% 5 == 0 || i == D_genes) {
                         cat(sprintf("Sim %03d – Gene %s (%d/%d)\n",
                                     sim, target_gene, i, D_genes),
                             file = file.path(data_dir, "bootstrap_progress.log"),
                             append = TRUE)
                       }
                     }
                     
                     end_time <- Sys.time()
                     duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
                     
                     cat(sprintf("Finished bootstrap %03d in %.2f sec\n", sim, duration),
                         file = file.path(data_dir, "bootstrap_progress.log"), append = TRUE)
                     
                     write.csv(R_hat, file = file.path(rdata_dir, sprintf("R_hat_%03d.csv", sim)), row.names = FALSE)
                     write.csv(xi_bs, file = file.path(xi_dir, sprintf("xi_%03d.csv", sim)), row.names = FALSE)
                     
                     return(list(R_hat = R_hat, xi = xi_bs))  
                   }

stopCluster(cl)

## ---------- aggregate ----------
R_mean  <- Reduce(`+`, lapply(results, `[[`, "R_hat"))  / num_simulations
xi_mean <- Reduce(`+`, lapply(results, `[[`, "xi"))     / num_simulations
xi_norm <- xi_mean / sqrt(sum(xi_mean^2))

write.csv(R_mean, file.path(data_dir, "R_gwps.csv"), row.names = FALSE)
write.csv(xi_norm, file.path(data_dir, "xi_gwps.csv"), row.names = FALSE)
#write.csv(R_mean, file.path(data_dir, "R_essential.csv"), row.names=FALSE)
#write.csv(xi_norm, file.path(data_dir, "xi_essential.csv"), row.names=FALSE)

