import argparse
import pandas as pd
import numpy as np
from helper_functions import notears_linear

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_path", required=True, help="Path to Y_matrix.csv")
    parser.add_argument("--out_path", required=True, help="Path to save output matrix")
    args = parser.parse_args()

    X = pd.read_csv(args.y_path).values
    W_est = notears_linear(X, lambda1=0.1, w_threshold=0.0)
    np.savetxt(args.out_path, W_est, delimiter=",")
