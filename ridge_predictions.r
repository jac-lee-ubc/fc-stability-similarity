## =====================================================================
## Ridge regression with feature selection and 10-fold CV.
##
## Expected CSV inputs:
##   - behaviour.csv  : ID and outcome variables
##   - covariates.csv : ID and Motion 
##   - cv_folds.csv   : ID and fold_reps
##   - FC files       : ID and features (edges)
## =====================================================================
library(data.table)
library(glmnet)
library(parallel)

## -----------------------------
## CONFIG
## -----------------------------
setwd("/path_to_folder")
id_col      <- "ID"
alpha_value <- 0            # alpha = 0 -> ridge regression

# Input files
behav_file  <- "behaviour.csv"
covar_file  <- "covariates.csv"
cv_file     <- "cv_folds.csv"
outcome_var <- "Age"        
motion_col  <- "Motion"
fc_files <- c("WB_edge_stability.csv", "WB_edge_similarity_Rest", "WB_edge_similarity_Movie")

# Number of CV repeitions (limit for debugging) 
max_reps_debug <- 100
metrics_to_use <- c("r", "MSE", "R2")

# Feature selection 
do_feature_select <- TRUE
fs_mode           <- "pvalue"   
fs_alpha          <- 0.05       # p-value threshold for feature selection

## -----------------------------
## Helper: performance metrics
## -----------------------------
compute_metrics <- function(y_true, y_pred, metrics = c("r","RMSE")) {
  out <- list()

  # Correlation 
  if ("r" %in% metrics) {
    out$r <- cor(y_true, y_pred, use = "complete.obs")
  }

  # RMSE 
    if ("RMSE" %in% metrics) {
    out$RMSE <- sqrt(mean((y_true - y_pred)^2))
  }

  unlist(out)
}

## -----------------------------
## Helper: partial correlation
## -----------------------------
partial_cor_motion_vec <- function(X_train_full, y_train, motion_train) {
  n <- length(y_train)

  # Design matrix 
  Z <- cbind(1, motion_train)
  ZZ_inv <- solve(t(Z) %*% Z)

  # Residualize X on motion 
  B_X <- ZZ_inv %*% (t(Z) %*% X_train_full)  
  R_X <- X_train_full - Z %*% B_X             

  # Residualize y on motion
  B_y <- ZZ_inv %*% (t(Z) %*% y_train)       
  r_y <- as.vector(y_train - Z %*% B_y)       

  # Mean-center residuals 
  R_Xc <- scale(R_X, center = TRUE, scale = FALSE)
  r_yc <- r_y - mean(r_y)

  # Partial correlation for each feature
  r_vec <- as.numeric(cor(R_Xc, r_yc))

  # p-values 
  valid <- !is.na(r_vec) & abs(r_vec) < 1
  t_vals <- rep(NA_real_, length(r_vec))
  p_vals <- rep(NA_real_, length(r_vec))

  t_vals[valid] <- r_vec[valid] * sqrt((n - 2) / (1 - r_vec[valid]^2))
  p_vals[valid] <- 2 * pt(-abs(t_vals[valid]), df = n - 2)
  p_vals[!valid & !is.na(r_vec)] <- 0

  list(r = r_vec, p = p_vals)
}

## =====================================================================
## LOAD INPUT (behaviour, covariates, CV folds)
## =====================================================================
behav_dt <- fread(behav_file)
covar_dt <- fread(covar_file)
cv_dt    <- fread(cv_file)

# Outcome and covariate vectors
y      <- behav_dt[[outcome_var]]
motion <- covar_dt[[motion_col]]

# Fold assignment matrix
fold_cols_all <- setdiff(names(cv_dt), id_col)
cv_mat_all    <- as.matrix(cv_dt[, ..fold_cols_all])

# Limit reps for debugging
n_rep_avail <- ncol(cv_mat_all)
n_rep_use   <- if (is.infinite(max_reps_debug)) n_rep_avail else min(max_reps_debug, n_rep_avail)
cv_mat    <- cv_mat_all[, seq_len(n_rep_use), drop = FALSE]
fold_cols <- fold_cols_all[seq_len(n_rep_use)]

## =====================================================================
## Run ridge prediction for each FC file
## =====================================================================

for (fc_file in fc_files) {

  cat("Processing FC file: ", fc_file, "\n", sep = "")

  # Load FC matrix (subjects x features)
  fc_dt  <- fread(fc_file)
  X_full <- as.matrix(fc_dt[, !id_col, with = FALSE])
  n <- length(y)
  p <- ncol(X_full)
  cat("X dims = ", n, " x ", p, "\n", sep = "")

  ## ---------------------------------------------------------
  ## For each rep: fit ridge on training folds & predict on test-fold
  ## ---------------------------------------------------------
  ridge_fun <- function(r) {

    folds_r  <- cv_mat[, r]
    fold_ids <- sort(unique(folds_r))
    K <- length(fold_ids)

    yhat <- rep(NA_real_, n)

    lambda_folds <- numeric(K)
    nfeat_folds  <- numeric(K)

    for (j in seq_along(fold_ids)) {

      f_id <- fold_ids[j]

      # Train/test split for this fold
      train_idx <- which(folds_r != f_id)
      test_idx  <- which(folds_r == f_id)

      X_train_full <- X_full[train_idx, , drop = FALSE]
      X_test_full  <- X_full[test_idx,  , drop = FALSE]

      y_train      <- y[train_idx]
      motion_train <- motion[train_idx]

      ## -----------------------------
      ## Feature selection 
      ## -----------------------------
      if (do_feature_select) {

        # Compute partial correlations controlling motion
        pc_res <- partial_cor_motion_vec(X_train_full, y_train, motion_train)

        # Keep features with p < fs_alpha 
        keep_idx <- which(pc_res$p < fs_alpha)

      } else {
        # Keep all features
        keep_idx <- seq_len(p)
      }

      nfeat_folds[j] <- length(keep_idx)

      # Reduced matrices after feature selection
      X_train <- X_train_full[, keep_idx, drop = FALSE]
      X_test  <- X_test_full[,  keep_idx, drop = FALSE]

      ## -----------------------------
      ## Choose lambda on training data
      ## -----------------------------
      cvfit <- cv.glmnet(
        x           = X_train,
        y           = y_train,
        alpha       = alpha_value,
        standardize = TRUE
      )

      lambda_min <- cvfit$lambda.min
      lambda_folds[j] <- lambda_min

      ## -----------------------------
      ## Fit ridge at lambda_min and predict held-out fold
      ## -----------------------------
      fit <- glmnet(
        x           = X_train,
        y           = y_train,
        alpha       = alpha_value,
        lambda      = lambda_min,
        standardize = TRUE
      )

      yhat[test_idx] <- as.numeric(predict(fit, newx = X_test, s = lambda_min))
    }

    list(
      pred       = yhat,
      lambda     = mean(lambda_folds, na.rm = TRUE),
      n_features = mean(nfeat_folds,  na.rm = TRUE)
    )
  }

  ## Run reps in parallel
  n_cores <- max(1L, detectCores() - 1L)
  ridge_results <- mclapply(seq_len(n_rep_use), ridge_fun, mc.cores = n_cores)

  ## Collect predictions + per-rep metadata
  pred_mat <- sapply(ridge_results, `[[`, "pred")              
  colnames(pred_mat) <- fold_cols
  lambda_used <- vapply(ridge_results, `[[`, numeric(1), "lambda")
  nfeat_used  <- vapply(ridge_results, `[[`, numeric(1), "n_features")

  ## Compute performance metrics for each rep
  obs_metrics_mat <- matrix(NA_real_, nrow = n_rep_use, ncol = length(metrics_to_use))
  colnames(obs_metrics_mat) <- metrics_to_use

  for (rr in seq_len(n_rep_use)) {
    m <- compute_metrics(y, pred_mat[, rr], metrics = metrics_to_use)
    obs_metrics_mat[rr, names(m)] <- m
  }

  obs_means <- colMeans(obs_metrics_mat, na.rm = TRUE)
  obs_sds   <- apply(obs_metrics_mat, 2, sd, na.rm = TRUE)

  cat("\nPerformance for ", outcome_var, " using ", fc_file, ":\n", sep = "")
  for (metric in metrics_to_use) {
    cat("  ", metric, ": mean = ", obs_means[metric], " (sd = ", obs_sds[metric], ")\n", sep = "")
  }

  ## ---------------------------------------------------------
  ## Save outputs
  ## ---------------------------------------------------------
  meta_dt <- data.table(
    repetition      = seq_len(n_rep_use),
    lambda_min_mean = lambda_used,
    n_features_mean = nfeat_used
  )
  meta_dt <- cbind(meta_dt, as.data.table(obs_metrics_mat))
  fwrite(meta_dt, paste0("ridge_meta_", outcome_var, "_", basename(fc_file)))

  pred_dt <- as.data.table(pred_mat)
  pred_dt[, ID := behav_dt[[id_col]]]
  setcolorder(pred_dt, c("ID", fold_cols))
  fwrite(pred_dt, paste0("ridge_pred_", outcome_var, "_", basename(fc_file)))
}

cat("\nAll FC files processed.\n")
