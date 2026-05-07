import numpy as np
import torch
from metrics.tsp import mi_assessment
from neural_networks.utils import create_dataloader_from_arrays, set_seed
from neural_networks.mlp import MLP
from preprocess_data import normalize_features, minmax_scale_features
from sklearn.metrics import mean_squared_error
import os

mode = "adamw"

#training config
device = torch.device("cuda")
n_seeds = 15
lr = 1e-4
weight_decay = 1e-3
n_epochs = 100
criterion = torch.nn.MSELoss()

#random seeds
seeds = np.random.default_rng(9).integers(0, 10000, size=n_seeds, dtype=int)

# train and test data
train_data = np.load("./array_data/sarcos/sarcos_inv.npz", allow_pickle=True)

x_train_arr = train_data['x']
y_train_arr = train_data['y']


test_data = np.load("./array_data/sarcos/sarcos_inv_test.npz", allow_pickle=True)
x_val_arr = test_data['x']
y_val_arr = test_data['y']

if mode == "overfitting":
    x_train_arr = x_train_arr[:1000]
    y_train_arr = y_train_arr[:1000]


y_min, y_max = np.min(y_train_arr), np.max(y_train_arr)
scale = y_max - y_min

y_train_arr, y_val_arr = y_train_arr.reshape(-1,1), y_val_arr.reshape(-1,1) 

x_train_arr, x_val_arr = normalize_features(x_train_arr, x_val_arr)
y_train_arr, y_val_arr = minmax_scale_features(y_train_arr, y_val_arr)

x_train = torch.tensor(x_train_arr, dtype = torch.float32, device = device)
y_train = torch.tensor(y_train_arr, dtype = torch.float32, device = device)

x_val = torch.tensor(x_val_arr, dtype = torch.float32, device = device)
y_val = torch.tensor(y_val_arr, dtype = torch.float32, device = device)
train_loader = create_dataloader_from_arrays(x_train_arr, y_train_arr, batch_size = 100, shuffle = True)


if __name__ == "__main__":
    final_path = "./experiments/early_stopping/"  
    os.makedirs(final_path,exist_ok=True)

    riv_full_ep = np.empty((n_seeds, x_val.shape[1], n_epochs))
    max_riv_val_ep = np.empty((n_seeds, n_epochs))
    rmse_val_ep = np.empty((n_seeds, n_epochs))
    rmse_train_ep = np.empty((n_seeds, n_epochs))

    for i, seed in enumerate(seeds):
        set_seed(seed)
        print("seed {}/{}".format(i+1, n_seeds))
        # model and optimizer definition
        model = MLP(input_size=x_train_arr.shape[1], output_size=y_train_arr.shape[1], activation=torch.nn.ReLU, hidden_sizes=[128]*3)
        if mode == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        elif mode == "sgd_2":
            optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, weight_decay=1e-4, nesterov=True, momentum = 0.999)
        elif mode == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr = 0.5e-4, weight_decay=1e-3, betas = (0.99, 0.999))
        else:    
            optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=weight_decay)
        best_rmse = float("inf")
        best_mi = float("inf")
        model = model.to(device)
        model.train()
        x_val = x_val.to(device)
        for ep in range(n_epochs):
            
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()*x.size(0)
            running_loss /= len(train_loader.dataset)  
            rmse_train_ep[i, ep] = np.sqrt(running_loss)*np.abs(np.max(y_train_arr)-np.min(y_train_arr))
            
            with torch.no_grad():
                model.eval()
                y_val_hat = model(x_val)
                y_val_hat = y_val_hat.cpu().detach().numpy()
                y_val_hat = y_val_hat*scale + y_min
                y_val_true = y_val_arr*scale + y_min
                y_val_hat = y_val_hat.reshape(-1, 1)
                #rmse_val = np.sqrt(np.mean((y_val_arr - y_val_hat)**2))
                rmse_val = mean_squared_error(y_val_true, y_val_hat)
                rmse_val = np.sqrt(rmse_val)
                rmse_val_ep[i, ep] = rmse_val
                riv = np.array([mi_assessment(x_val_arr[:, i].reshape(-1,1), y_val_true, y_val_hat) for i in range(x_val_arr.shape[1])])
                riv_full_ep[i, :, ep] = riv
                max_riv_val_ep[i, ep] = np.max(riv)
            if np.max(riv) < best_mi:
                best_mi = np.max(riv)
                torch.save(model.state_dict(), os.path.join(final_path, f"best_mi_model_{mode}_seed_{i}.pth"))
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                torch.save(model.state_dict(), os.path.join(final_path, f"best_rmse_model_{mode}_seed_{i}.pth"))
            print("epoch {}/{}, rmse: {}, mi: {}".format(ep+1, n_epochs, rmse_val, np.max(riv)))
    np.save(f"{final_path}/riv_full_{mode}_val.npy", riv_full_ep)        
    np.save(f"{final_path}/max_riv_{mode}_val.npy", max_riv_val_ep)   
    np.save(f"{final_path}/rmse_{mode}_train.npy", rmse_train_ep)
    np.save(f"{final_path}/rmse_{mode}_val.npy", rmse_val_ep)        

