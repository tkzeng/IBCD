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

generate_data_inhibition <- function(G, N_cont, N_int, int_beta=-2, noise='gaussian', normalize=TRUE){
  D <- nrow(G)
  int_sizes <- rep(N_int, D) # rnbinom(D, mu=N_int - N_int/10, size=N_int/10) + N_int/10

  N_int_cs <- cumsum(c(0, int_sizes))
  XB <- matrix(0, nrow=N_int_cs[length(N_int_cs)], ncol=D)
  
  #for(d in 1:D){
  #  start = N_int_cs[d]
  #  end = N_int_cs[d+1]
  #  XB[(start+1):end, d] = 1
  #}
  
  if (N_int > 0) {
    for(d in 1:D){
      start = N_int_cs[d]
      end = N_int_cs[d+1]
      if (start < end) {
        XB[(start+1):end, d] = 1
      }
    }
  }
  
  net_vars <- colSums(G**2)
  eps_vars <- max(0.9, max(net_vars)) - net_vars + 0.1
  total_var <- net_vars + eps_vars
  
  if(noise == 'gaussian'){
    eps_cont <- t(matrix(rnorm(D*N_cont, sd=sqrt(eps_vars)), nrow=D, ncol=N_cont))
    eps_int <- t(matrix(rnorm(D*sum(int_sizes), sd=sqrt(eps_vars)), nrow=D, ncol=sum(int_sizes)))
  } else{
    stop('NotImplementedError')
  }
  Y_cont <- eps_cont %*% solve(diag(D) - G)
  if(normalize){
    # This mimics perturb-seq normalization
    mu_cont <- colMeans(Y_cont)
    sd_cont <- apply(Y_cont, 2, sd)
    
    Y_cont <- t((t(Y_cont) - mu_cont)/sd_cont)
    Y_int <- (t(t(XB) * int_beta * sd_cont) + eps_int) %*% solve(diag(D) - G)
    Y_int <- t((t(Y_int) - mu_cont)/sd_cont)
    
    # Also need to normalize the graph
    R <- get_tce(get_observed(G), normalize=sd_cont)
    G <- get_direct(R)$G
    # TODO: update eps_vars
  } else{
    Y_int <- (t(t(XB) * int_beta) + eps_int) %*% solve(diag(D) - G)
    R <- get_tce(get_observed(G))
  }
  Y <- rbind(Y_cont, Y_int)
  
  colnames(Y) <- paste0("V", 1:D)
  targets <- c(rep("control", N_cont), paste0("V", rep(1:D, times=int_sizes)))
  return(list(Y = Y, targets = targets, G = G, R = R, int_beta=int_beta, eps_vars=eps_vars))
}


generate_dataset <- function(D, N_cont, N_int, int_beta=-2,
                              graph = 'scalefree', v = 0.2, p = 0.4,
                              DAG = FALSE, C = floor(0.1*D), noise = 'gaussian',
                              model = 'inhibition'){
  G <- generate_network(D, graph, p, v, DAG)
  if(C > 0){
    if(graph == "scalefree"){
      new_vars <- matrix(mc2d::rpert(D*C, 0.01, v^(log(D)/log(log(D))), 2*v), nrow=C)
    } else if(graph == "random"){
      new_vars <- matrix(mc2d::rpert(D*C, 0.01, v^(log(D)), 2*v), nrow=C)
    }
    
    G <- cbind(rbind(G, new_vars), matrix(0, nrow=D+C, ncol=C))
  }
  if(model == 'inhibition'){
    data = generate_data_inhibition(G, N_cont, N_int, int_beta, noise)
  } else if (model == 'knockout'){
    data = generate_data_knockout(G, N_cont, N_int, noise)
  }
  
  Y <- data$Y
  G <- data$G
  var_all <- apply(Y %*% G, 2, var)[1:D]
  var_obs <- apply(Y[,1:D] %*% G[1:D, 1:D], 2, var)
  if(C > 0){
    var_conf <- apply(Y[,(D+1):(D+C)] %*% G[(D+1):(D+C), 1:D], 2, var)
  } else {
    var_conf <- rep(0, length=D)
  }
  var_eps <- 1-var_all
  
  return(list(Y=Y[1:(N_cont+N_int*D), 1:D], targets=data$targets[1:(N_cont+N_int*D)], R=data$R[1:D, 1:D],
              G=G[1:D, 1:D], var_all=var_all, var_obs=var_obs,
              var_conf=var_conf, var_eps=var_eps, int_beta=data$int_beta, eps_vars=data$eps_vars))
}


base_dir <- file.path(getwd(), "obs")

int_beta_default <- -2
v_default <- 0.25
graph_type <- "scalefree" # "scalefree" #"random"
DAG_flag <- TRUE
C_default <- 0
N_int_default <- 0
variables_50d <- c(5, 15, 25, 50, 75, 100)

# Replicate seeds
replicate_seeds <- 42:51  # 10 replicates

run_one <- function(D, variable, N_int = N_int_default, p, seed, out_dir,
                    int_beta = int_beta_default,
                    v = v_default,
                    graph = graph_type,
                    DAG = DAG_flag,
                    C = C_default) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf("[D=%d | N_int=%d | p=%.3f | seed=%d] -> %s",
                  D, N_int, p, seed, out_dir))
  set.seed(seed)
  
  N_cont <- 2 * D * variable
  
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
  
}


### scale free ###
D_50 <- 50
p_50 <- 0.066

for (variable in variables_50d) {
  for (seed in replicate_seeds) {
    out_dir <- file.path(base_dir, "50d", as.character(variable), "sf", as.character(seed))
    run_one(D = D_50, variable = variable, p = p_50, seed = seed, out_dir = out_dir)
  }
}

# -----------------------------
D_150 <- 150
p_150 <- 0.108
variable <- 100

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "150d", "sf", as.character(seed))
  run_one(D = D_150, variable = variable, p = p_150, seed = seed, out_dir = out_dir)
}

# -----------------------------
D_250 <- 250
p_250 <- 0.124
variable <- 100

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "250d", "sf", as.character(seed))
  run_one(D = D_250, variable = variable, p = p_250, seed = seed, out_dir = out_dir)
}

# -----------------------------
D_500 <- 500
p_500 <- 0.139
variable <- 100

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "500d", "sf", as.character(seed))
  run_one(D = D_500, variable = variable, p = p_500, seed = seed, out_dir = out_dir)
}

####################################
### random ###
####################################

D_50 <- 50
p_50 <- 0.1

for (variable in variables_50d) {
  for (seed in replicate_seeds) {
    out_dir <- file.path(base_dir, "50d", as.character(variable), "random", as.character(seed))
    run_one(D = D_50, variable = variable, p = p_50, seed = seed, out_dir = out_dir)
  }
}

# -----------------------------
D_150 <- 150
p_150 <- 0.033
variable <- 100

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "150d", "random", as.character(seed))
  run_one(D = D_150, variable = variable, p = p_150, seed = seed, out_dir = out_dir)
}

# -----------------------------
D_250 <- 250
p_250 <- 0.02
variable <- 100

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "250d", "random", as.character(seed))
  run_one(D = D_250, variable = variable, p = p_250, seed = seed, out_dir = out_dir)
}

# -----------------------------
D_500 <- 500
p_500 <- 0.01
variable <- 100

for (seed in replicate_seeds) {
  out_dir <- file.path(base_dir, "500d", "random", as.character(seed))
  run_one(D = D_500, variable = variable, p = p_500, seed = seed, out_dir = out_dir)
}

