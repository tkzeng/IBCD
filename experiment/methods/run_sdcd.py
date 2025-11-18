import argparse
import pandas as pd
import torch
from sdcd.models import SDCD
from sdcd.utils import create_intervention_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_path", required=True, help="Path to Y_matrix.csv")
    parser.add_argument("--targets_path", required=True, help="Path to targets.txt")
    parser.add_argument("--out_path", required=True, help="Path to save adjacency matrix")
    args = parser.parse_args()

    # ---- Read inputs
    r_df = pd.read_csv(args.y_path)
    with open(args.targets_path, "r") as f:
        targets = [line.strip() for line in f]

    # ---- Attach perturbation label column
    r_df["perturbation_label"] = ["obs" if t == "control" else t for t in targets]

    # ---- Build SDCD dataset and train
    X_dataset = create_intervention_dataset(r_df, perturbation_colname="perturbation_label")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = SDCD()
    model.train(X_dataset, finetune=True, device=device)

    # ---- Get adjacency and save
    adj_pred = model.get_adjacency_matrix(threshold=True)
    pd.DataFrame(adj_pred).to_csv(args.out_path, index=False)
