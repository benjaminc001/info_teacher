from argparse import ArgumentParser
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from preprocess_data import normalize_features, minmax_scale_features
from neural_networks.mlp import MLP
from neural_networks.resnet import ResNetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from neural_networks.ft_transformer import FTTransformer
from neural_networks.test import test
from neural_networks.trainer import trainer
from metrics.tsp import mi_assessment as tsp
from neural_networks.utils import set_seed, save_experiment_results
from datetime import datetime
import os
from torch import nn

def quantize_y_quantiles(y, n_quantiles=15):
    bins = np.quantile(y, np.linspace(0, 1, n_quantiles + 1))
    idx = np.digitize(y, bins) - 1
    idx = np.clip(idx, 0, n_quantiles - 1)
    centers = np.array([y[idx == i].mean() for i in range(n_quantiles)])
    return centers[idx], centers

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
def build_linear_model():
    model =LinearRegression()
    return model
def build_knn_model():
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')
    return model
def build_transformer(args):
    input_size = args.get('n_features', 10)
    output_size = args.get('output_size', 1)
    model = FTTransformer(n_features=input_size, output_dim=output_size)
    return model
def build_gbdt_model(random_seed=0):
    model = HistGradientBoostingRegressor(learning_rate=0.05, random_state=random_seed, 
                                          max_iter=80, max_depth=None, l2_regularization=1e-3, 
                                          early_stopping=True, n_iter_no_change=15, min_samples_leaf=50, validation_fraction=0.1,)
    return model
def build_rf_model(random_seed=0):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=10,bootstrap=True, random_state=random_seed)
    return model
def build_decision_tree_model(random_seed=0):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(splitter="random",max_depth=None, min_samples_leaf=10, random_state=random_seed)
    return model
def train_sklearn_model(model, x_train, y_train, val_data=None):
    if val_data is not None:
        x_val, y_val = val_data
        y_val = y_val.ravel()
        y_train = y_train.ravel()
        model.fit(x_train, y_train, X_val = x_val, y_val=y_val)
    else:
        y_train = y_train.ravel()
        model.fit(x_train, y_train)
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
parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)")
parser.add_argument("--network_name", type=str, default="shallow", help="Name of the neural network architecture")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
parser.add_argument("--base_lr", type=float, default=1e-4, help="Base learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay for the optimizer")
parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use for training")
args = parser.parse_args()


if args.network_name.lower() in ["linear", "knn"]:
    args.n_seeds = 1  # only one seed for non-neural network models
    args.learning_rate = 0.0  # learning rate not used for sklearn models
    args.patience = None  # no early stopping for sklearn models
    args.num_epochs = 1  # only one epoch needed for sklearn models
    args.device = "cpu"  # sklearn models run on CPU
    args.optimizer = None  # no optimizer for sklearn models
seeds = np.random.default_rng(args.seed).integers(0, 10000, size=args.n_seeds, dtype=int)
if args.dataset_name in ["sarcos", "unfavorable_sarcos"]:
    data_train = np.load(f"./array_data/sarcos/sarcos_inv.npz", allow_pickle=True)
    data_test = np.load(f"./array_data/sarcos/sarcos_inv_test.npz", allow_pickle=True)
    x = data_train['x']
    y = data_train['y'] 
    X_train = data_train['x']
    y_train = data_train['y']
    X_test = data_test['x']
    y_test = data_test['y']
    oracle = None
else:
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
arg_snn = {
        'input_size': x.shape[1],
        'hidden_sizes': [128, 128, 128],
        'output_size': y.shape[1],
        'activation': nn.SELU,
    }

arg_resnet_network = {
        'input_size': x.shape[1],
        'output_size': y.shape[1],
        'num_blocks': 5,
        'hidden_size': 1024,
        'normalization':'layer',
    }  


arg_transformer = {
    'n_features': x.shape[1],
    'output_size': y.shape[1],
}

def build_network(**kwargs):
    if args.network_name == "shallow":
        arg_mlp_network['hidden_sizes'] = [32]
    elif args.network_name == "deep":
        arg_mlp_network['hidden_sizes'] = [128]*3
    elif args.network_name == "overparameterized":
        arg_mlp_network['hidden_sizes'] = [1024]*6
    elif args.network_name == "large":
        arg_mlp_network['hidden_sizes'] = [2048]*3
    elif args.network_name == "small":
        arg_mlp_network["hidden_sizes"] = [5]
    model = build_mlp_model(arg_mlp_network)
    if args.network_name == "resnet":
        model = build_resnet_model(arg_resnet_network)
    if args.network_name == "linear":          
        model = build_linear_model()
    if args.network_name == "snn":
        model = build_mlp_model(arg_snn)
    if args.network_name == "knn":
        model = build_knn_model()
    if args.network_name == "gbdt":          
        model = build_gbdt_model(kwargs.get('random_seed', 0))
    if args.network_name == "transformer":
        model = build_transformer(arg_transformer)
    if args.network_name == "rf":
        model = build_rf_model(kwargs.get('random_seed', 0))
    if args.network_name == "decision_tree":
        model = build_decision_tree_model(kwargs.get('random_seed', 0))
    return model

if __name__ == "__main__":
    oracle_train = None
    oracle_min, oracle_max = None, None
    if oracle is not None:
        X_train, X_test, y_train, y_test, oracle_train, oracle_test = train_test_split(x, y, oracle, test_size=args.test_size, random_state=412)
    else:
        if args.dataset_name not in ["sarcos", "unfavorable_sarcos"]:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=412)
    if args.dataset_name in ["unfavorable_housing", "unfavorable_housing_2", "noisy_ccpp", "unfavorable_sarcos", "unfavorable_housing_full"]:
        if args.dataset_name == "unfavorable_housing" or args.dataset_name == "unfavorable_housing_2" or args.dataset_name == "unfavorable_housing_full":
            n_quantiles = 2
        elif args.dataset_name == "unfavorable_sarcos":
            n_quantiles = 3
        else:
            n_quantiles = 3
        y_train, _ = quantize_y_quantiles(y_train, n_quantiles=n_quantiles)
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
            if args.network_name in ["linear", "gbdt", "rf", "decision_tree", "knn"]:
                if args.network_name == "gbdt":
                    model = build_gbdt_model(seed)
                    model = train_sklearn_model(model, x_train_subset,  y_train_subset,)
                elif args.network_name == "rf":
                    model = build_rf_model(seed)
                    model = train_sklearn_model(model, x_train_subset,  y_train_subset)
                elif args.network_name == "decision_tree":
                    model = build_decision_tree_model(seed)
                    model = train_sklearn_model(model, x_train_subset,  y_train_subset)
                else:
                    model = train_sklearn_model(model, x_train_subset,  y_train_subset)
                prediction = model.predict(X_test)
                prediction = prediction.reshape(-1,1)
            else:    
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
                prediction_unnormalized = prediction * (y_max - y_min) + y_min
                oracle_metric = np.mean((oracle_test - prediction_unnormalized)**2)
                results["oracle"][j, k] = oracle_metric 
            results["mi"][j, k] = mi
            results["mse"][j, k] = mse
            results["r2"][j, k] = r2
            results["riv"][j, :, k] = riv
            results["max_riv"][j, k] = max_riv
            print("mutual information:", mi)
            if oracle is not None: print("oracle mean squared error:", oracle_metric)
    
    
    snapshot_path = f"./experiments/{args.dataset_name}/{args.network_name}"
    os.makedirs(snapshot_path, exist_ok = True)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.network_name in ["linear", "gbdt"]:
        config_file = os.path.join(snapshot_path, f'config_{args.network_name}_{date}.txt')
    else:
        config_file = os.path.join(snapshot_path, f'config_{args.network_name}_{args.optimizer}_{date}.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)
    results_path = f"./experiments/{args.dataset_name}/{args.network_name}/"
    if args.network_name in ["linear", "gbdt", "knn", "rf", "decision_tree"]:
        filename = os.path.join(results_path, f"results_{args.dataset_name}_{args.network_name}.npz") 
    else:
        filename = os.path.join(results_path, f"results_{args.optimizer}_{args.dataset_name}_{args.network_name}.npz")
    os.makedirs(results_path, exist_ok=True)
    save_experiment_results(results, filename)