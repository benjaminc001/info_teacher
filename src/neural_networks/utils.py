import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def build_optimizer_fcn(model, optimizer_name='adam', kwargs=None):
    if optimizer_name.lower() == 'adam':
        def optimizer_fcn(model):
            return optim.Adam(model.parameters(), lr=kwargs.get('lr', 1e-4), weight_decay=kwargs.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'sgd':
        def optimizer_fcn(model):
            return optim.SGD(model.parameters(), lr=kwargs.get('lr', 1e-4), weight_decay=kwargs.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'adamw':
        def optimizer_fcn(model):
            return optim.AdamW(model.parameters(), lr=kwargs.get('lr', 1e-4), weight_decay=kwargs.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'rmsprop':
        def optimizer_fcn(model):
            return optim.RMSprop(model.parameters(), lr=kwargs.get('lr', 1e-4), weight_decay=kwargs.get('weight_decay', 1e-5))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer_fcn(model)


def create_dataloader_from_arrays(x, y, batch_size=32, shuffle=True):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_experiment_config(args, path):
    import json
    with open(path, 'w') as f:
        json.dump(args, f, indent=4)

def save_experiment_results(results, path):
    np.savez_compressed(path, **results)