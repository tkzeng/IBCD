multiple_iv_reg <- function(target, .X, .targets){
  inst_obs <- .targets == target
  control_obs <- .targets == "control"
  n_target <- sum(inst_obs)
  n_control <- sum(control_obs)
  n_total <- n_target + n_control
  X_target <- .X[inst_obs, ]
  X_control <- .X[control_obs, ]
  this_feature <- which(colnames(.X) == target)
  
  # Combine target and control data
  X_combined <- rbind(X_target, X_control)
  
  # Extract the predictor for the target variable
  X_exp <- X_combined[, this_feature]
  
  # Perform Ordinary Least Squares (OLS) Regression
  # Predict the target variable using all predictors, including itself
  data_reg <- as.data.frame(X_combined)
  names(data_reg) <- colnames(.X)
  data_reg$target <- X_combined[, target]
  
  # Construct the regression formula
  formula_reg <- as.formula(paste(target, "~", paste(colnames(.X), collapse = " + ")))
  
  # Fit the regression model
  model <- lm(formula_reg, data = data_reg)
  
  # Extract coefficients and standard errors
  beta_hat <- coef(model)
  se_hat <- summary(model)$coefficients[, "Std. Error"]
  
  # Rename coefficients to include "beta_hat" and "se_hat" prefixes
  names(beta_hat) <- paste(names(beta_hat), "beta_hat", sep="_")
  names(se_hat) <- paste(names(se_hat), "se_hat", sep="_")
  
  # Combine the results into a single data frame
  result <- c(list(target = target, beta_obs = NA), as.list(beta_hat), as.list(se_hat))
  
  return(as.data.frame(result))
}