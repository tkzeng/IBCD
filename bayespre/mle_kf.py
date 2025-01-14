import numpy as np
import zipfile
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

def iterative_mle(folder_path, max_iter=100, tol=1e-4, eps_psd=1e-8):
    # Load R matrices from the zip file
    R_matrices = []
    with zipfile.ZipFile(folder_path, 'r') as z:
        for filename in sorted(z.namelist()):
            if filename.endswith(".csv"):
                with z.open(filename) as f:
                    R = pd.read_csv(f).values
                    R_matrices.append(R)

    R_matrices = np.array(R_matrices) 
    N, p, q = R_matrices.shape
    R_bar = np.mean(R_matrices, axis=0)

    # Initialize U (row covariance) and V (column covariance)
    U = np.cov(R_bar.T) 
    V = np.cov(R_bar.T) 

    for iteration in tqdm(range(max_iter)):
        U_prev = U.copy()
        V_prev = V.copy()

        # Update V (column covariance, \Sigma_s)
        V_temp = np.zeros((q, q))
        U_inv = np.linalg.inv(U_prev + eps_psd * np.eye(q))  # Use U_prev for update
        for R in R_matrices:
            diff = (R - R_bar)
            V_temp += diff @ U_inv @ diff.T 
        V = V_temp / (q * N)
        
        # Update U (row covariance, \Sigma_c)
        U_temp = np.zeros((p, p))
        V_inv = np.linalg.inv(V + eps_psd * np.eye(p))  # Use V_prev for update
        for R in R_matrices:
            diff = (R - R_bar)
            U_temp += diff.T @ V_inv @ diff  
        U = U_temp / (p * N)

    return U, V

def kron_factorization(S, D, lr=1e-1, max_iter=5000, tol=1e-8):
    S_tensor = torch.tensor(S, dtype=torch.float32)
    U = torch.randn(D, D, requires_grad=True)
    V = torch.randn(D, D, requires_grad=True)
    optimizer = optim.Adam([U, V], lr=lr)

    def loss_fn(S_approx, S_true):
        return torch.norm(S_true - S_approx, p='fro') / torch.norm(S_true, p='fro')

    for iteration in tqdm(range(max_iter)):
        optimizer.zero_grad()
        S_approx = torch.kron(U, V)
        loss = loss_fn(S_approx, S_tensor)
        loss.backward()
        optimizer.step()
        relative_error = torch.norm(S_tensor - S_approx) / torch.norm(S_tensor)
        if relative_error.item() < tol:
            break

    return U.detach().numpy(), V.detach().numpy()


def is_positive_definite(matrix, tol=0):
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues > tol)


if __name__ == "__main__":
    # Load data
    S = pd.read_csv('S_hat_10x10matrix.csv', index_col=0).values
    folder_path = "R_hat_files.zip"

    # Perform Kronecker Approximation using iterative_mle
    U_est, V_est = iterative_mle(folder_path)
    S_kron_est = np.kron(U_est, V_est)

    is_U_psd = is_positive_definite(U_est)
    is_V_psd = is_positive_definite(V_est)
    print(f"Is U positive definite? {is_U_psd}")
    print(f"Is V positive definite? {is_V_psd}")

    # Perform Kronecker Approximation using kron_factorization
    D = 10
    U_pytorch, V_pytorch = kron_factorization(S, D)
    S_kron_pytorch = np.kron(U_pytorch, V_pytorch)

    is_U_pytorch_psd = is_positive_definite(U_pytorch)
    is_V_pytorch_psd = is_positive_definite(V_pytorch)
    print(f"Is U_pytorch positive definite? {is_U_pytorch_psd}")
    print(f"Is V_pytorch positive definite? {is_V_pytorch_psd}")

    frobenius_norm_est = np.linalg.norm(S - S_kron_est, 'fro')  
    mae_est = np.mean(np.abs(S - S_kron_est))  
    relative_error_est = frobenius_norm_est / np.linalg.norm(S, 'fro')  

    frobenius_norm_pytorch = np.linalg.norm(S - S_kron_pytorch, 'fro') 
    mae_pytorch = np.mean(np.abs(S - S_kron_pytorch)) 
    relative_error_pytorch = frobenius_norm_pytorch / np.linalg.norm(S, 'fro')  

    print("\nMetrics for Kronecker Approximation (iterative_mle):")
    print(f"Frobenius Norm: {frobenius_norm_est:.6f}, MAE: {mae_est:.6e}, Relative Error: {relative_error_est:.6e}")

    print("\nMetrics for Kronecker Approximation (kron_factorization):")
    print(f"Frobenius Norm: {frobenius_norm_pytorch:.6f}, MAE: {mae_pytorch:.6e}, Relative Error: {relative_error_pytorch:.6e}")

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(S, cmap='viridis')
    plt.title("Original Matrix (S)")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(S_kron_est, cmap='viridis')
    plt.title("Iterative MLE Approximation")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(S_kron_pytorch, cmap='viridis')
    plt.title("Kronecker Factorization (KF)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()