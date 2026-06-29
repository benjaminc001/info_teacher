from argparse import ArgumentParser
from neural_networks.utils import create_dataloader_from_arrays, set_seed
from neural_networks.mlp import MLP
import os
import pickle
import numpy as np
import torch
from metrics.tsp import mi_assessment as tsp
from torch import optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error

parser = ArgumentParser()
parser.add_argument("--mode", default = "vanilla", type = str, help = "Mode for early stopping, could be 'vanilla' or 'overfitting'")
parser.add_argument("--criterion", default = "rmse", type = str, help = "Early stopping Criterion, could be 'rmse' or 'rim'")
parser.add_argument("--patience", default = 10, type = int, help = "Number of epochs with no improvement")
parser.add_argument("--device", default = "cpu", type = str, help = "Device to perform the neural networks training, could be 'cpu' or 'cuda'")
parser.add_argument("--lr", default = 1e-3, type = float, help = "Optimizer learning rate")
parser.add_argument("--num_epochs", default = 100, type = int, help = "Total number of epochs")

args = parser.parse_args()

mode = args.mode
device = args.device
lr = args.lr
n_epochs = args.num_epochs
patience = args.patience
criterion = args.criterion

n_split_seeds = 3
n_param_seeds = 21

split_seeds = np.random.default_rng(5).integers(2_001, 3_000, size = n_split_seeds, dtype = int)
param_seeds = np.random.default_rng(10).integers(0, 2_000, size = n_param_seeds, dtype = int)
final_path = os.path.join("./experiments/early_stopping_cv/", criterion)


def train_epoch(dataloader, model, loss_fn, optimizer, device = "cpu"):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        #print(pred.shape, y.shape)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return np.sqrt(total_loss / len(dataloader))

def validation(model, X_Arr, y_Arr, train_scaler, device = "cpu"):
    # Pone el modelo en modo de evaluación (apaga Dropout y BatchNorm)
    model.eval()
    # Desactiva el cálculo de gradientes para ahorrar memoria y velocidad
    with torch.no_grad():
        y_hat = model(torch.tensor(X_Arr, device = device, dtype=torch.float32)).detach().to("cpu").numpy()
    y_hat_transformed = train_scaler.inverse_transform(y_hat)
    rmse = root_mean_squared_error(y_Arr, y_hat_transformed)
    riv = np.array([tsp(X_Arr[:, i].reshape(-1,1), y_hat_transformed, y_Arr) for i in range(X_Arr.shape[1])])
    rim = np.max(riv)
    return rmse, rim

def main():
    models_path = os.path.join(final_path, "models")
    os.makedirs(final_path, exist_ok = True)
    os.makedirs(models_path, exist_ok = True)
    data_path = "./array_data/sarcos/sarcos_inv.npz"
    test_data_path = "./array_data/sarcos/sarcos_inv_test.npz"
    train_data = np.load(data_path, allow_pickle = True)
    test_data = np.load(test_data_path, allow_pickle = True)
    
    x_train_total = train_data["x"]
    y_train_total = train_data["y"]

    x_test = test_data["x"]
    y_test = test_data["y"]

    y_test = y_test.reshape(-1, 1)
    epochs = range(n_epochs)

    results = {}

    rim_test_ = np.zeros((n_param_seeds, n_split_seeds))
    rmse_test_ = np.zeros((n_param_seeds, n_split_seeds))
    epochs_counter = np.zeros((n_param_seeds, n_split_seeds))

    timeseries_split = TimeSeriesSplit(n_splits = n_split_seeds, test_size = 2_000)
    for i, (train_index, val_index) in enumerate(timeseries_split.split(x_train_total)):
        x_train, x_val = x_train_total[train_index], x_train_total[val_index]
        y_train, y_val = y_train_total[train_index], y_train_total[val_index]
        print(f"Split Seed ({i + 1} / {n_split_seeds})")
        print(f"train data length: {len(x_train)}, val data length: {len(x_val)}")
        y_train, y_val = y_train.reshape(-1, 1), y_val.reshape(-1, 1)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_val = sc.transform(x_val)
        x_test_scaled = sc.transform(x_test)
        min_max_sc = MinMaxScaler()
        y_train = min_max_sc.fit_transform(y_train)
        
        results[i] = {}
        if mode == "overfitting":
            x_train, y_train = x_train[:1_000, :], y_train[:1_000, :]
        train_loader = create_dataloader_from_arrays(x_train, y_train, batch_size=100)

        for j, seed in enumerate(param_seeds):
            set_seed(seed)
            rmse_train_list = []
            rmse_val_list = []
            rim_val_list = []
            model = MLP(x_train_total.shape[1], hidden_sizes=[128]*3, output_size=y_train_total.shape[1], activation=torch.nn.ReLU)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            min_loss = float("inf")
            ep_no_improve = 0
            for ep in epochs:
                if (ep + 1)%10 == 0:
                    print(f"seed {j+1}/{len(param_seeds)} // ep {ep + 1}/{len(epochs)}")
                rmse_train =  train_epoch(dataloader = train_loader, model = model, loss_fn = torch.nn.MSELoss(), optimizer = optimizer, device = device)
                rmse_val, rim_val =  validation(model = model, X_Arr = x_val, y_Arr = y_val, train_scaler = min_max_sc, device = device)
                best_model_path = f"mode_{mode}_best_{criterion}_split_seed_{i}_seed_{j}.pth"

                final_model_path = os.path.join(models_path, best_model_path)
                if criterion == "rmse":
                    if rmse_val < min_loss:
                        min_loss = rmse_val
                        ep_no_improve = 0
                        best_ep = ep
                        torch.save(model.state_dict(), final_model_path)
                    else:
                        ep_no_improve += 1
                else:
                    if rim_val < min_loss:
                        min_loss = rim_val
                        ep_no_improve = 0
                        best_ep = ep
                        torch.save(model.state_dict(), final_model_path)
                    else:
                        ep_no_improve += 1
                if ep_no_improve >= patience:
                    print(f"training finished at epoch {ep + 1}")
                    break
                rmse_train_list.append(rmse_train)
                rmse_val_list.append(rmse_val)
                rim_val_list.append(rim_val)
            state_dict = torch.load(final_model_path)
            model.load_state_dict(state_dict)    
            rmse_test, rim_test = validation(model=model, X_Arr=x_test_scaled, y_Arr=y_test, train_scaler=min_max_sc, device=device)
            
            rmse_test_[j, i] = rmse_test
            rim_test_[j, i] = rim_test
            epochs_counter[j, i] = ep_no_improve
            results[i][j] = {"train_loss": rmse_train_list, "val_loss": rmse_val_list, "val_rim": rim_val_list, "n_eps": len(rim_val_list), "best_ep": best_ep}
            print(f"rmse test: {rmse_test}, rim test: {rim_test}, best epoch: {best_ep}, best {criterion}: {min_loss}")
    return results, rmse_test_, rim_test_, epochs_counter

if __name__ == "__main__":
    results, rmse_test, rim_test, epochs_counter = main()
    results_saving_pkl = os.path.join(final_path, f"mode_{mode}_criterion_{criterion}.pkl")
    results_saving_path = os.path.join(final_path, f"mode_{mode}_criterion_{criterion}.npz")
    np.savez_compressed(results_saving_path, rmse_test = rmse_test, rim_test = rim_test, epochs_count = epochs_counter)
    with open(results_saving_pkl, "wb") as f:
        pickle.dump(results, f)
    print("Files saved successfully!")
