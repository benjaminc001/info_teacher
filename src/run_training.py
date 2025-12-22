from argparse import ArgumentParser
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from preprocess_data import normalize_features, minmax_scale_features
from neural_networks.mlp import MLP
from neural_networks.resnet import ResNetRegressor
from neural_networks.linear_model import LinearModel
from neural_networks.ft_transformer import FTTransformer
from neural_networks.test import test
from neural_networks.trainer import trainer
from metrics.tsp import mi_assessment as tsp
from neural_networks.utils import set_seed, save_experiment_results
from datetime import datetime
import os
from torch import nn

def build_mlp_model(args):
    input_size = args.get('input_size', 10)
    hidden_sizes = args.get('hidden_sizes', [64, 64])
    output_size = args.get('output_size', 1)
    activation = args.get('activation', nn.ReLU)

    model = MLP(input_size, hidden_sizes, output_size, activation)
    return model

def build_resnet_model(args):
    input_size = args.get('input_size', 10)
    output_size = args.get('output_size', 1)
    num_blocks = args.get('num_blocks', 3)
    hidden_size = args.get('hidden_size', 64)

    model = ResNetRegressor(input_size, output_size, num_blocks, hidden_size)
    return model
def build_linear_model(args):
    input_size = args.get('input_size', 10)
    output_size = args.get('output_size', 1)
    num_units = args.get('n_units', 10)

    model = LinearModel(input_size, output_size, num_units)
    return model
def build_transformer(args):
    input_size = args.get('n_features', 10)
    output_size = args.get('output_size', 1)
    model = FTTransformer(n_features=input_size, output_dim=output_size)
    return model
def none_or_int(x):
    if x.lower() == "none":
        return None
    else:
        return int(x)

os.makedirs("./experiments", exist_ok=True)
os.makedirs("./models", exist_ok=True)

parser = ArgumentParser()
parser.add_argument("--dataset_name", default="ccpp",type=str, help="Name of the dataset to use for training")
parser.add_argument("--model_path", type=str, default="./models/", help="Path to save the trained model")
parser.add_argument("--n_seeds", type=int, default=8, help="Number of different random seeds to run experiments with")
parser.add_argument("--patience", type=none_or_int, default=10, help="Patience for early stopping")
parser.add_argument("--test_size", type=float, default=2_000, help="Number of samples to use for testing")
parser.add_argument("--n_points", type=int, default=30, help="Number of data points to generate for experiment")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cpu or cuda)")
parser.add_argument("--network_name", type=str, default="shallow", help="Name of the neural network architecture")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
parser.add_argument("--base_lr", type=float, default=1e-4, help="Base learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay for the optimizer")
parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use for training")
args = parser.parse_args()

seeds = np.random.default_rng(args.seed).integers(0, 10000, size=args.n_seeds, dtype=int)

loaded_data = np.load(f"./array_data/{args.dataset_name}/{args.dataset_name}.npz", allow_pickle=True)
x = loaded_data['x']
y = loaded_data['y']
oracle = loaded_data.get('oracle', None)


arg_mlp_network = {
        'input_size': x.shape[1],
        'hidden_sizes': [64, 64],
        'output_size': y.shape[1],
        'activation': nn.ReLU,
    }
arg_resnet_network = {
        'input_size': x.shape[1],
        'output_size': y.shape[1],
        'num_blocks': 3,
        'hidden_size': 64,
        'normalization':'layer',
    }  

arg_linear_model =  {
    'input_size': x.shape[1],
    'output_size': y.shape[1],
    'n_units' : 1
}

arg_transformer = {
    'n_features': x.shape[1],
    'output_size': y.shape[1],
}

def build_network():
    if args.network_name == "shallow":
        arg_mlp_network['hidden_sizes'] = [32]
    elif args.network_name == "deep":
        arg_mlp_network['hidden_sizes'] = [128]*3
    elif args.network_name == "overparameterized":
        arg_mlp_network['hidden_sizes'] = [1024]*6

    elif args.network_name == "small":
        arg_mlp_network["hidden_sizes"] = [2]
    model = build_mlp_model(arg_mlp_network)
    if args.network_name == "resnet":
        model = build_resnet_model(arg_resnet_network)
    if args.network_name == "linear":
        model = build_linear_model(arg_linear_model)
    if args.network_name == "transformer":
        model = build_transformer(arg_transformer)
    return model

if __name__ == "__main__":
    oracle_train = None
    oracle_min, oracle_max = None, None
    if oracle is not None:
        X_train, X_test, y_train, y_test, oracle_train, oracle_test = train_test_split(x, y, oracle, test_size=args.test_size, random_state=412)
    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=412)
    if args.dataset_name == "noisy_ccpp":
        y_train = y_train + np.random.default_rng(8).normal(0.0, 2.5, size=y_train.shape)
    y_max, y_min = np.max(y_train), np.min(y_train)
    if oracle_train is not None:
        oracle_max, oracle_min = np.max(oracle_train), np.min(oracle_train)
    X_train, X_test = normalize_features(X_train, X_test)
    y_train, y_test = minmax_scale_features(y_train, y_test)
    training_size = np.logspace(np.log10(256), np.log10(X_train.shape[0]), num=args.n_points, dtype=int)
    results = {"mi":np.empty((args.n_seeds, len(training_size))), 
               "mse":np.empty((args.n_seeds, len(training_size))), 
               "max_riv": np.empty((args.n_seeds, len(training_size))),
               "riv": np.empty((args.n_seeds, X_test.shape[1], len(training_size))),
               "r2": np.empty((args.n_seeds, len(training_size))), 
               "oracle": np.zeros((args.n_seeds, len(training_size))),
               "x": training_size, "n_seeds": args.n_seeds}
    for j,seed in enumerate(seeds):
        set_seed(seed)
        print(f"Seed {j+1}/{len(seeds)}")
        mi_results = np.empty(args.n_points)
        riv_results = np.empty(args.n_points)
        mse_results = np.empty(args.n_points)
        r2_results = np.empty(args.n_points)
        riv_total = np.empty((X_test.shape[1], args.n_points))
        for k, n in enumerate(training_size):
            x_train_subset = X_train[:n]
            y_train_subset = y_train[:n]

  
            arrays = {
                'x_train': x_train_subset,
                'y_train': y_train_subset,
                'x_val': X_test,
                'y_val': y_test,
                'oracle_train': None,
                'oracle_val': None
            }

            train_args = {
                'device': args.device,
                'model_path': args.model_path,
                'network_name': args.network_name,
                'dataset_name': args.dataset_name,
                'y_min': y_min,
                'y_max': y_max,
                'seed': seed,
                'patience': args.patience,
                'num_epochs': args.num_epochs,
                'base_lr': args.base_lr,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
                'optimizer': args.optimizer
            }

            oracle_args = {
                'oracle_max': oracle_max,
                'oracle_min': oracle_min
            }
            model = build_network()
            x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(args.device)
            trained_model, metrics = trainer(train_args, model, arrays)
            trained_model.eval()
            with torch.no_grad():
                prediction = trained_model(x_test_tensor).detach().cpu().numpy()
            
            riv = np.array([tsp(X_test[:, i].reshape(-1,1), y_test, prediction) for i in range(X_test.shape[1])])
            max_riv = np.max(riv)
            arrs_test = {"x_test": X_test, "y_test": y_test, "prediction": prediction,}
            mse, mi, r2 = test(train_args, arrs_test)
            if oracle is not None:
                oracle_metric = np.mean((prediction - oracle_test)**2)
                results["oracle"][j, k] = oracle_metric 
            results["mi"][j, k] = mi
            results["mse"][j, k] = mse
            results["r2"][j, k] = r2
            results["riv"][j, :, k] = riv
            results["max_riv"][j, k] = max_riv
    
    
    snapshot_path = f"./experiments/{args.dataset_name}/{args.network_name}"
    os.makedirs(snapshot_path, exist_ok = True)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = os.path.join(snapshot_path, f'config_{args.network_name}_{args.optimizer}_{date}.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)
    results_path = f"./experiments/{args.dataset_name}/{args.network_name}/"
    filename = os.path.join(results_path, f"results_{args.optimizer}_{args.dataset_name}_{args.network_name}.npz")
    os.makedirs(results_path, exist_ok=True)
    save_experiment_results(results, filename)