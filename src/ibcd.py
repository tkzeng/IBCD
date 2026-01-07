import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import cvxpy as cp
from numpyro import infer
from numpyro.infer import MCMC, NUTS

from empirical_prior import (
    scale_free_degree,
    solve_edge_weights_rowwise,
    load_R_and_SE_hat,
    empirical_bayes_em,
    solve_spike_slab_diagonal_spike,
)
from model import matrix_model_spike_horseshoe, compute_lfsr
from model_lowrank import matrix_model_lowrank
from feature_prior import (
    load_gene_features,
    fit_lowrank_regression,
    fit_lowrank_classification,
    compute_lowrank_sparsity,
)
from iv_regression import xi_norm, run_all_IV


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data)

    xi_norm(df, out_dir=args.output_dir)

    #print("2) Running 2SLS IV regressions...")
    run_all_IV(df, out_dir=args.output_dir)

    # 2SLS outputs
    Rhat_path = f"{args.output_dir}/R.csv"
    SE_hat_path = f"{args.output_dir}/SE_hat.csv"
    xi_path = f"{args.output_dir}/xi.csv"
    U_path = f"{args.output_dir}/U.csv"
    V_path = f"{args.output_dir}/V.csv"

    Rhat_df = pd.read_csv(Rhat_path)
    colnames = Rhat_df.columns.tolist()

    xi = pd.read_csv(xi_path).values
    U_mat = pd.read_csv(U_path)
    V_mat = pd.read_csv(V_path)

    D = xi.shape[0]

    dag_flag = (args.dag.lower() == "true")

    if args.prior.lower() == "sf":
        # -------- Scale-free (SF) prior --------
        print("2) Using SF prior (Scale-Free)...")
        R = Rhat_df.values
        pi0_mat = scale_free_degree(R)
        offdiag_mask = ~np.eye(D, dtype=bool)
        pi0_i = pi0_mat[offdiag_mask].reshape(D, D-1).mean(axis=1)

        print("Estimated spike weight:", pi0_i)
        print("3) Running edge specific weights for SF...")
        pi0_ij, pi_k_ij = solve_edge_weights_rowwise(
            xi,
            pi0_i,
            dag=dag_flag,
            alpha_sf=args.alpha_sf,
            solver=cp.ECOS,
        )

        use_lowrank = False

    elif args.prior.lower() == "er":
        # -------- Erdős–Rényi (ER) prior --------
        print("2) Using ER prior (Erdős–Rényi)...")
        # Load off-diagonal or upper-tri entries depending on DAG
        w, se_hat = load_R_and_SE_hat(Rhat_path, SE_hat_path, dag=dag_flag)
        #print(f"Shape of w: {w.shape}, Shape of se: {se_hat.shape}")
        pi0, pi_slabs, slab_scales = empirical_bayes_em(
            w,
            se_hat,
            alpha_er=args.alpha_er,
        )
        print("Estimated spike weight:", float(pi0))
        #print("Sum pi:", float(pi0 + pi_slabs.sum()))

        print("3) Running edge specific weights for ER...")
        pi0_ij, pi_k_ij, _ = solve_spike_slab_diagonal_spike(xi, pi0=pi0)

        use_lowrank = False

    elif args.prior.lower() == "lowrank":
        # -------- Low-Rank Feature-Based prior --------
        print("2) Using Low-Rank prior...")

        if args.prior_dir is not None:
            # Load pre-computed priors from fit_lowrank_prior.py
            print(f"   Loading pre-computed priors from {args.prior_dir}/")
            pi0_A = np.load(f"{args.prior_dir}/pi0_A.npy")
            pi0_B = np.load(f"{args.prior_dir}/pi0_B.npy")

            # Infer rank from loaded priors
            args.rank = pi0_A.shape[1]
            print(f"   Loaded pi0_A: {pi0_A.shape}, pi0_B: {pi0_B.shape}")
            print(f"   Inferred rank K = {args.rank}")

        elif args.features is not None:
            # Fit priors inline
            print("   Fitting priors from gene features...")
            X_features = load_gene_features(args.features)
            R_hat = Rhat_df.values
            SE_hat = pd.read_csv(SE_hat_path).values

            print(f"   Loaded gene features: {X_features.shape[0]} genes x {X_features.shape[1]} features")
            print(f"   Rank K = {args.rank}")

            if args.fit_mode == "regression":
                print("3) Fitting low-rank model (regression mode)...")
                W_A, W_B, A_fitted, B_fitted = fit_lowrank_regression(
                    X_features, R_hat, K=args.rank, lambda_reg=args.lambda_reg
                )
            else:
                print("3) Fitting low-rank model (classification mode)...")
                W_A, W_B, A_fitted, B_fitted = fit_lowrank_classification(
                    X_features, R_hat, SE_hat, K=args.rank, lambda_reg=args.lambda_reg
                )

            # Compute sparsity priors from fitted A, B
            # Use EM to learn target sparsity from data (like ER prior)
            if args.sparsity_scale is None:
                print("   Learning sparsity level via EM...")
                w, se_hat_flat = load_R_and_SE_hat(Rhat_path, SE_hat_path, dag=dag_flag)
                pi0_global, _, _ = empirical_bayes_em(w, se_hat_flat, alpha_er=args.alpha_er)
                print(f"   EM estimated global sparsity: {float(pi0_global):.3f}")
                pi0_A, pi0_B = compute_lowrank_sparsity(A_fitted, B_fitted, target_sparsity=float(pi0_global))
            else:
                pi0_A, pi0_B = compute_lowrank_sparsity(A_fitted, B_fitted, scale=args.sparsity_scale)

            # Save fitted weights for interpretability
            pd.DataFrame(W_A, columns=[f"K{k}" for k in range(args.rank)]).to_csv(
                f"{args.output_dir}/W_A.csv", index=False
            )
            pd.DataFrame(W_B, columns=colnames).to_csv(
                f"{args.output_dir}/W_B.csv", index=False
            )
            pd.DataFrame(A_fitted, columns=[f"K{k}" for k in range(args.rank)], index=colnames).to_csv(
                f"{args.output_dir}/A_fitted.csv", index=True
            )
            pd.DataFrame(B_fitted, columns=colnames).to_csv(
                f"{args.output_dir}/B_fitted.csv", index=False
            )
        else:
            raise ValueError("--prior lowrank requires either --features or --prior_dir")

        use_lowrank = True

    else:
        raise ValueError("args.prior must be 'sf', 'er', or 'lowrank'.")

    U_lower = jnp.linalg.cholesky(jnp.array(U_mat.values))
    V_lower = jnp.linalg.cholesky(jnp.array(V_mat.values))

    print("4) Running inference...")

    if use_lowrank:
        # Low-rank MCMC model
        kernel = NUTS(
            matrix_model_lowrank,
            target_accept_prob=0.7,
            max_tree_depth=10,
            init_strategy=infer.init_to_median(num_samples=3),
        )

        mcmc = MCMC(
            kernel,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            progress_bar=True,
        )

        mcmc.run(
            jax.random.PRNGKey(42),
            obs_data=Rhat_df.values,
            pi0_A=jnp.array(pi0_A),
            pi0_B=jnp.array(pi0_B),
            U_lower=U_lower,
            V_lower=V_lower,
            D=D,
            K=args.rank,
            epsilon=args.epsilon_reg,
        )
    else:
        # Standard full-rank MCMC model
        kernel = NUTS(
            matrix_model_spike_horseshoe,
            target_accept_prob=0.7,
            max_tree_depth=10,
            init_strategy=infer.init_to_median(num_samples=3),
        )

        mcmc = MCMC(
            kernel,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            progress_bar=True,
        )

        mcmc.run(
            jax.random.PRNGKey(42),
            obs_data=Rhat_df.values,
            pi0_ij=pi0_ij,
            U_lower=U_lower,
            V_lower=V_lower,
            D=D,
        )


    # ------------------------ outputs ---------------------------------
    posterior_G = np.asarray(
        jax.device_get(mcmc.get_samples(group_by_chain=True)["G"]),
        dtype=np.float32,
    )
    np.save(os.path.join(args.output_dir, "G_draws.npy"), posterior_G)

    flat_G = posterior_G.reshape(-1, D, D)

    posterior_mean = flat_G.mean(axis=0)
    pd.DataFrame(posterior_mean, columns=colnames, index=colnames).to_csv(
        f"{args.output_dir}/G.csv", index=True
    )

    pip_matrix = (np.abs(flat_G) > args.epsilon).mean(axis=0)
    pd.DataFrame(pip_matrix, columns=colnames, index=colnames).to_csv(
        f"{args.output_dir}/pip.csv", index=True
    )

    lfsr_matrix = compute_lfsr(flat_G)
    pd.DataFrame(lfsr_matrix, columns=colnames, index=colnames).to_csv(
        f"{args.output_dir}/lfsr.csv", index=True
    )

    # Save A, B posterior means for low-rank model
    if use_lowrank:
        posterior_A = np.asarray(
            jax.device_get(mcmc.get_samples(group_by_chain=True)["A"]),
            dtype=np.float32,
        )
        posterior_B = np.asarray(
            jax.device_get(mcmc.get_samples(group_by_chain=True)["B"]),
            dtype=np.float32,
        )

        flat_A = posterior_A.reshape(-1, D, args.rank)
        flat_B = posterior_B.reshape(-1, args.rank, D)

        A_mean = flat_A.mean(axis=0)
        B_mean = flat_B.mean(axis=0)

        pd.DataFrame(A_mean, columns=[f"K{k}" for k in range(args.rank)], index=colnames).to_csv(
            f"{args.output_dir}/A.csv", index=True
        )
        pd.DataFrame(B_mean, columns=colnames).to_csv(
            f"{args.output_dir}/B.csv", index=False
        )

        np.save(os.path.join(args.output_dir, "A_draws.npy"), posterior_A)
        np.save(os.path.join(args.output_dir, "B_draws.npy"), posterior_B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "IBCD pipeline.\n"
            "1) Load data.csv (observation + intervention)\n"
            "2) Run 2SLS\n"
            "3) Choose SF (scale-free) or ER (Erdős–Rényi) empirical prior\n"
            "4) Fit empirical Bayesian spike-and-slab prior on matrix normal model\n"
            "5) Output G, PIP, and LFSR.\n"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to input data CSV.",
    )

    parser.add_argument(
        "--prior",
        required=True,
        choices=["sf", "er", "lowrank"],
        help="Choice of empirical prior: 'sf' = scale-free, 'er' = Erdős–Rényi, 'lowrank' = feature-based low-rank.",
    )

    parser.add_argument(
        "--features",
        default=None,
        help="Path to gene features CSV (D genes x F features). Used with --prior lowrank.",
    )

    parser.add_argument(
        "--prior_dir",
        default=None,
        help="Path to directory with pre-computed priors from fit_lowrank_prior.py. Used with --prior lowrank.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=10,
        help="Rank K for low-rank factorization G=AB. Default = 10.",
    )

    parser.add_argument(
        "--fit_mode",
        choices=["regression", "classification"],
        default="regression",
        help="Fitting mode for low-rank prior: 'regression' fits to R_hat, 'classification' fits to significant edges. Default = regression.",
    )

    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.1,
        help="L1 regularization strength for low-rank fitting. Default = 0.1.",
    )

    parser.add_argument(
        "--sparsity_scale",
        type=float,
        default=None,
        help="Scale factor for converting fitted A,B to sparsity priors. If not set, uses EM to learn sparsity level from data.",
    )

    parser.add_argument(
        "--epsilon_reg",
        type=float,
        default=1e-5,
        help="Regularization epsilon for matrix inversion stability: (I - G + eps*I)^{-1}. Default = 1e-5.",
    )

    parser.add_argument(
        "--dag",
        required=True,
        choices=["true", "false"],
        help="'true' = allow Directed Acyclic Graph, 'false' = allow Cyclic Graph.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save all outputs.",
    )

    parser.add_argument(
        "--alpha_sf",
        type=float,
        default=1.0,
        help="Penalty parameter for SF prior (scale-free row-wise optimization). Default=1.0.",
    )

    parser.add_argument(
        "--alpha_er",
        type=float,
        default=2.0,
        help="Alpha for EM in ER prior. Controls shrinkage strength. Default=2.0.",
    )

    parser.add_argument(
        "--num_warmup",
        type=int,
        default=300,
        help="Number of NUTS warm-up iterations. Default = 300.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of posterior samples per chain after warm-up. Default = 1000.",
    )

    parser.add_argument(
        "--num_chains",
        type=int,
        default=3,
        help="Number of parallel MCMC chains. Default = 3.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help=(
            "Threshold for computing PIP: edges with |G| > epsilon are counted as active. "
            "Default = 0.05."
        ),
    )

    args = parser.parse_args()
    main(args)
