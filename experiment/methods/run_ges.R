library(pcalg)

args <- commandArgs(trailingOnly = TRUE)
y_path  <- args[1]  
out_path <- args[2] 

Y <- as.matrix(read.csv(y_path, header = TRUE))

score <- new("GaussL0penObsScore", Y)
ges_res <- ges(score)
G_hat <- as(ges_res$essgraph, "matrix") + 0.0

write.csv(G_hat, out_path, row.names = FALSE)
