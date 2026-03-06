import numpy as np
import torch
from neural_networks.mlp import MLP
from neural_networks.trainer import trainer

n_seeds = 8
train_data = np.load("array_data/sarcos_inv.npz", allow_pickle=True)

x_train_arr = train_data['x']
y_train_arr = train_data['y']

test_data = np.load("array_data/sarcos_inv_test.npz", allow_pickle=True)
x_val_arr = test_data['x']
y_val_arr = test_data['y']
arrays = {
    'x_train': x_train_arr,
    'y_train': y_train_arr,
    'x_val': x_val_arr,
    'y_val': y_val_arr,
}
model = MLP(input_dim=x_train_arr.shape[1], output_dim=y_train_arr.shape[1], hidden_dims=[128, 128])
args = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': "./models",
    'network_name': 'MLP',
    'dataset_name': 'sarcos_inv',
    'num_epochs': 100,
}
if __name__ == "__main__":

    pass