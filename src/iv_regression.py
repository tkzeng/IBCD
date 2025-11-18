import pandas as pd
import numpy as np
from tqdm import tqdm

def xi_norm(df, out_dir="."):
    """
    Given a dataframe 'df' loaded from data.csv
    (feature columns + 'target' column),
    compute xi_norm and save it to xi_true.csv.
    """

    # Extract feature columns
    feature_cols = [c for c in df.columns if c != "target"]
    Y_np = df[feature_cols].to_numpy(dtype=np.float64)
    mask_np = (df["target"] == "control").to_numpy()

    Y_C = Y_np[mask_np]
    # Sigma = t(Y_C) %*% Y_C / nrow(Y_C)
    n = Y_C.shape[0]
    Sigma = (Y_C.T @ Y_C) / n
    xi = Sigma ** 2
    np.fill_diagonal(xi, 0.0)
    xi_norm_ = xi / np.sqrt(np.sum(xi**2))

    #out_df = pd.DataFrame(xi_norm_, columns=feature_cols)
    #out_df.to_csv(out_path, index=False)
    pd.DataFrame(xi_norm_, columns=feature_cols).to_csv(f"{out_dir}/xi.csv", index=False)


def multiple_iv_reg_UV(target, X_df, targets):

    X = X_df.to_numpy(dtype=np.float64)
    targets = np.asarray(targets)

    # Step 1: identify samples
    inst_obs = (targets == target)
    control_obs = (targets == "control")

    n_target = inst_obs.sum()
    n_control = control_obs.sum()
    n_total = n_target + n_control

    # Step 2: split X
    X_target = X[inst_obs, :]
    X_control = X[control_obs, :]

    # Step 3: which feature index corresponds to this target
    feature_index = list(X_df.columns).index(target)

    # Step 4: exposure columns
    X_exp_target = X_target[:, feature_index]
    X_exp_control = X_control[:, feature_index]

    # Step 5: beta_inst_obs
    beta_inst_obs = X_exp_target.sum() / n_target

    # Step 6: beta_hat = colSums(X_target) / X_target[, target_index] sum
    sums_target = X_target.sum(axis=0)
    beta_hat = sums_target / sums_target[feature_index]

    # Step 7: residuals (rbind of target residuals and control residuals)
    resid_target = X_target - np.outer(X_exp_target, beta_hat)
    resid_control = X_control - np.outer(X_exp_control, beta_hat)
    resid = np.vstack([resid_target, resid_control])

    # Step 8: V_hat
    V_hat = (resid.T @ resid) / resid.shape[0]

    # Step 9: se_hat
    se_hat = np.sqrt(np.diag(V_hat) / (n_target * (beta_inst_obs ** 2)))

    colnames = X_df.columns.tolist()
    beta_hat_named = {f"{col}_beta_hat": val for col, val in zip(colnames, beta_hat)}
    se_hat_named = {f"{col}_se_hat": val for col, val in zip(colnames, se_hat)}

    beta_se = {
        "target": target,
        "beta_obs": beta_inst_obs,
        **beta_hat_named,
        **se_hat_named
    }

    return {
        "beta_se": beta_se,
        "U_i": 1.0 / (n_target * beta_inst_obs**2),
        "V_hat": V_hat
    }

def run_all_IV(df, out_dir="."):
    feature_cols = [c for c in df.columns if c != "target"]
    X_df = df[feature_cols]
    targets = df["target"].tolist()
    D = len(feature_cols)

    genes = feature_cols

    U_diag = np.zeros(D)
    V_sum = np.zeros((D, D))
    R_hat = np.zeros((D, D))
    SE_hat = np.zeros((D, D))

    for i, gene in enumerate(tqdm(genes, desc="1) Running IV for all targets...")):
        res = multiple_iv_reg_UV(gene, X_df, targets)

        U_diag[i] = res["U_i"]
        V_sum += res["V_hat"]

        # extract *_beta_hat in order
        beta_values = [res["beta_se"][f"{g}_beta_hat"] for g in genes]
        se_values   = [res["beta_se"][f"{g}_se_hat"] for g in genes]

        R_hat[i, :] = beta_values
        SE_hat[i, :] = se_values

    U = np.diag(U_diag)
    V = V_sum / D

    # Save CSVs
    pd.DataFrame(R_hat, columns=genes).to_csv(f"{out_dir}/R.csv", index=False)
    pd.DataFrame(SE_hat, columns=genes).to_csv(f"{out_dir}/SE_hat.csv", index=False)
    pd.DataFrame(U, columns=genes).to_csv(f"{out_dir}/U.csv", index=False)
    pd.DataFrame(V, columns=genes).to_csv(f"{out_dir}/V.csv", index=False)


