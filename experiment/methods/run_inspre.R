library(inspre)

args <- commandArgs(trailingOnly = TRUE)

y_path      <- args[1]
targets_path <- args[2]
out_path     <- args[3]

Y <- as.matrix(read.csv(y_path))
targets <- readLines(targets_path)

result <- fit_inspre_from_X(
  X=Y, targets=targets, filter=FALSE, cv_folds=5,
  verbose=1, rho=10, constraint="UV", DAG=TRUE, ncores=4
)

best_index <- which.min(result$eps_hat_G)
G_hat <- result$G_hat[,,best_index]
write.csv(G_hat, out_path, row.names=FALSE)
