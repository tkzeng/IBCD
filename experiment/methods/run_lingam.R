library(pcalg)

args <- commandArgs(trailingOnly = TRUE)

y_path      <- args[1]   
out_path    <- args[2]  

Y <- as.matrix(read.csv(y_path))
lingam_res <- pcalg::lingam(Y)
G_hat <- t(lingam_res$Bpruned)

write.csv(G_hat, out_path, row.names=FALSE)
