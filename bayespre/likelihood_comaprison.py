import numpy as np
import zipfile
import pandas as pd
import matplotlib.pyplot as plt


def estimate_U_V(folder_path):
    # Step 1: Read all R^ matrices
    R_matrices = []
    with zipfile.ZipFile(folder_path, 'r') as z:
        for filename in sorted(z.namelist()):
            if filename.endswith(".csv"):
                with z.open(filename) as f:
                    R = pd.read_csv(f).values
                    R_matrices.append(R)
    
    R_matrices = np.array(R_matrices)  # Shape: (N, D, D)
    N, D, _ = R_matrices.shape

    # Step 2: Compute the mean matrix R_bar
    R_bar = np.mean(R_matrices, axis=0)  # Shape: (D, D)

    # Step 3: Compute U and V
    U = np.zeros((D, D))
    V = np.zeros((D, D))

    for R in R_matrices:
        diff = R - R_bar
        U += diff @ diff.T
        V += diff.T @ diff

    # Normalize U and V
    U /= (N * np.trace(V))
    V /= (N * np.trace(U))

    return U, V

def low_rank_approximation(U, singular_values, Vt, rank):
    # Use the top 'rank' singular values and vectors
    S_approx = np.dot(U[:, :rank], np.dot(np.diag(singular_values[:rank]), Vt[:rank, :]))
    return S_approx

def is_positive_definite(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)  
    return np.all(eigenvalues > 0)


folder = "/Users/seongwoohan/Desktop/inspre_bayes/data/S_hat_matrix.csv"
df = pd.read_csv(folder, index_col=0)
S = df.values
print("Shape of S:", S.shape)

# Perform SVD
U_svd, singular_values, Vt = np.linalg.svd(S)


ranks = [10, 50, 60, 80, 90]
approximations = {}
metrics = {}

for rank in ranks:
    S_approx = low_rank_approximation(U_svd, singular_values, Vt, rank)
    approximations[rank] = S_approx
    
    frobenius_norm = np.linalg.norm(S - S_approx, 'fro')
    mae = np.mean(np.abs(S - S_approx))
    relative_error = frobenius_norm / np.linalg.norm(S, 'fro')

    is_pd = is_positive_definite(S_approx)
    
    metrics[rank] = {'Frobenius Norm': frobenius_norm, 
                     'MAE': mae, 
                     'Relative Error': relative_error,
                     'Positive Definite': is_pd,
                     }

# Compute Kronecker approximation using U and V
folder_path = "/Users/seongwoohan/Desktop/inspre_bayes/data/R_hat_files.zip"
U, V = estimate_U_V(folder_path)
S_kron = np.kron(U, V)

# Compute metrics for Kronecker approximation
frobenius_norm_kron = np.linalg.norm(S - S_kron, 'fro')
mae_kron = np.mean(np.abs(S - S_kron))
relative_error_kron = frobenius_norm_kron / np.linalg.norm(S, 'fro')


is_pd_kron = is_positive_definite(S_kron)


metrics['Kronecker MLE'] = {
    'Frobenius Norm': frobenius_norm_kron,
    'MAE': mae_kron,
    'Relative Error': relative_error_kron,
    'Positive Definite': is_pd_kron,
}

# Print metrics
print("\nMetrics for Low-Rank Approximations and Kronecker Approximation:")
for rank, metric in metrics.items():
    print(f"{rank}: Frobenius Norm = {metric['Frobenius Norm']:.6f}, "
          f"MAE = {metric['MAE']:.6e}, Relative Error = {metric['Relative Error']:.6e}, "
          f"Positive Definite = {metric['Positive Definite']}")

plt.figure(figsize=(15, 25))
plt.subplot(6, 3, 1)
plt.imshow(S, cmap='viridis')
plt.title("Original Matrix")
plt.colorbar()

for i, rank in enumerate(ranks):
    # Low-rank approximation
    plt.subplot(6, 3, 2 + i * 3)
    plt.imshow(approximations[rank], cmap='viridis')
    plt.title(f"Low-Rank Approx (Rank = {rank})")
    plt.colorbar()

    # Difference
    plt.subplot(6, 3, 3 + i * 3)
    plt.imshow(S - approximations[rank], cmap='coolwarm', vmin=-np.max(np.abs(S - approximations[rank])), vmax=np.max(np.abs(S - approximations[rank])))
    plt.title(f"Difference (Rank = {rank})")
    plt.colorbar()

# Kronecker approximation and its difference at the bottom
plt.subplot(6, 3, 17)
plt.imshow(S_kron, cmap='viridis')
plt.title("Kronecker MLE")
plt.colorbar()

plt.subplot(6, 3, 18)
plt.imshow(S - S_kron, cmap='coolwarm', vmin=-np.max(np.abs(S - S_kron)), vmax=np.max(np.abs(S - S_kron)))
plt.title("Difference (Kronecker)")
plt.colorbar()

plt.tight_layout()
plt.savefig('low_rank_and_kronecker_comparison.pdf')  # Save the figure
plt.show()