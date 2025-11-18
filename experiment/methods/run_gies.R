library(pcalg)

args <- commandArgs(trailingOnly = TRUE)
y_path      <- args[1]   
targets_path <- args[2] 
out_path     <- args[3]  

Y <- as.matrix(read.csv(y_path))
target_strings <- readLines(targets_path)

targets <- lapply(target_strings, function(x) {
  if (x == "control") integer(0) else as.integer(sub("V", "", x))
})
target_labels <- sapply(targets, function(x) paste0(sort(x), collapse = ","))
unique_target_keys <- unique(target_labels)
unique_targets <- lapply(unique_target_keys, function(k) {
  if (k == "") integer(0) else as.integer(strsplit(k, ",")[[1]])
})
target.index <- match(target_labels, unique_target_keys)

score_obj <- new("GaussL0penIntScore", data = Y,
                 targets = unique_targets, target.index = target.index)
gies_res <- gies(score_obj)
G_hat <- as(gies_res$essgraph, "matrix") + 0.0

write.csv(G_hat, out_path, row.names = FALSE)
