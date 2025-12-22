import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--optimizer_name", default="sgd")
parser.add_argument("--network_name", default="small")
parser.add_argument("--dataset_name", default="ccpp")
parser.add_argument("--set_log_mse", default=True)

args = parser.parse_args()

network_name_base = str.lower(args.network_name)
dataset_name_base = str.lower(args.dataset_name)
results_path = f"./experiments/{dataset_name_base}/{network_name_base}/results_{args.optimizer_name}_{dataset_name_base}_{network_name_base}.npz"
saving_path_root = f"./experiments/figures/{dataset_name_base}/"
os.makedirs(saving_path_root, exist_ok=True)
saving_path = os.path.join(saving_path_root, f"{dataset_name_base}_{network_name_base}_fig.pdf")


is_synthetic = (dataset_name_base not in ["sarcos", "ccpp", "housing", "reduced_housing", "noisy_ccpp", "unfavorable_housing"])

def generate_figure(results_path, saving_path, set_log_mse=True):

    metrics_dict = {"mi": "Mutual Information", 
                    "mse":"MSE", 
                    "max_riv":"R.I.V", 
                    "r2": "$R^2$", 
                    "oracle": "Ground Truth error"}
    
    results_dict = np.load(results_path, allow_pickle = True)
    metrics = list(results_dict.keys())
    n_seeds = results_dict["n_seeds"]
    x = results_dict["x"]
    removed_keys = ["x", "n_seeds", "riv"]
    for el in removed_keys: metrics.remove(el)
    if not is_synthetic:
        metrics.remove("oracle")
    fig, ax = plt.subplots(1, len(metrics))
    fig.set_size_inches((6*len(metrics), 4))
    for i, metric in enumerate(metrics):
        for j in range(n_seeds):
            ax[i].plot(x, results_dict[metric][j, :])
        ax[i].set_title(metrics_dict[metric])
        ax[i].grid()
        if metric == "mse":
            if set_log_mse:
                ax[i].set_yscale("log")
    plt.tight_layout()
    plt.savefig(saving_path)

if __name__ == "__main__":
    generate_figure(results_path, saving_path, args.set_log_mse)

    
