import os
import torch
import numpy as np
from torch import nn
from torch import amp
from tqdm import tqdm
from neural_networks.test import test
from copy import deepcopy
from neural_networks.utils import create_dataloader_from_arrays, build_optimizer_fcn

def trainer(args, model, arrays):
    # ---------- Config ----------
    device = args.get('device', 'cpu')
    model_path = args.get('model_path', "./models")
    network_name = args.get('network_name', 'MLP')
    dataset_name = args.get('dataset_name', 'default_dataset')
    y_min, y_max = args.get('y_min', None), args.get('y_max', None)
    oracle_min, oracle_max = args.get('oracle_min', None), args.get('oracle_max', None)
    seed = args.get('seed', 0)
    patience = args.get('patience', None)
    num_epochs = args.get('num_epochs', 100)
    base_lr = args.get('base_lr', 1e-3)
    weight_decay = args.get('weight_decay', 0.0)
    batch_size = args.get('batch_size', 32)

    scaler = amp.GradScaler()
    optimizer_name = args.get('optimizer', 'adamw')
    optimizer = build_optimizer_fcn(
        model,
        optimizer_name,
        kwargs=args.get('optimizer_kwargs',
                        {'lr': base_lr, 'weight_decay': weight_decay})
    )

    criterion = nn.MSELoss()

    # ---------- Data ----------
    x_train_arr = arrays.get('x_train')
    y_train_arr = arrays.get('y_train')
    x_val_arr = arrays.get('x_val')
    y_val_arr = arrays.get('y_val')
    oracle_train_arr = arrays.get('oracle_train')
    oracle_val_arr = arrays.get('oracle_val')
    train_dataloader = create_dataloader_from_arrays(
        x_train_arr, y_train_arr, batch_size=batch_size, shuffle=True, seed=seed)

    # ---------- Metrics ----------
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_mi": [],
        "val_mi": [],
        "train_r2": [],
        "val_r2": [],
        "train_oracle_loss": [],
        "val_oracle_loss": [],
        "best_epoch": None,
        "best_val_loss": None,
    }
    
    # ---------- Checkpoint ----------
    ckpt_path = (
        f"{model_path}{dataset_name}/"
    )
    os.makedirs(ckpt_path, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    # ---------- Training loop ----------
    epochs = tqdm(range(num_epochs), desc="Training", ncols=90)
    model.to(device)
    for epoch in epochs:

        # ===== TRAIN =====
        model.train()
        running_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with amp.autocast(device_type=device):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()*inputs.size(0)
        train_epoch_loss = running_loss / len(train_dataloader.dataset)
        train_epoch_loss = train_epoch_loss*(y_max - y_min)**2 
        metrics["train_loss"].append(train_epoch_loss)

        # ===== VALIDATION =====
        model.eval()
        x_train_tensor = torch.tensor(x_train_arr, dtype=torch.float32, device=device) if x_train_arr is not None else None
        x_val_tensor = torch.tensor(x_val_arr, dtype=torch.float32, device=device) if x_val_arr is not None else None
        with torch.no_grad():
            training_prediction = model(x_train_tensor).detach().cpu().numpy()
            val_prediction = model(x_val_tensor).detach().cpu().numpy() 
        
        args_test = {"y_min": y_min, "y_max": y_max}
        arrs_train = {'x_test': x_train_arr, 'y_test': y_train_arr, 'prediction': training_prediction}
        arrs_val = {'x_test': x_val_arr, 'y_test': y_val_arr, 'prediction': val_prediction}
        _, train_mi, r2_train = test(args_test, arrs_train)
        val_epoch_loss, val_mi, r2_val = test(args_test, arrs_val)
        if oracle_train_arr is not None:
            oracle_train_arr = (oracle_train_arr - oracle_min) / (oracle_max - oracle_min)
            oracle_train_loss = np.mean((oracle_train_arr - training_prediction)**2)*(oracle_max - oracle_min)**2
            metrics["train_oracle_loss"].append(oracle_train_loss)
        if oracle_val_arr is not None:
            oracle_val_arr = (oracle_val_arr - oracle_min) / (oracle_max - oracle_min)
            oracle_val_loss = np.mean((oracle_val_arr - val_prediction)**2)*(oracle_max - oracle_min)**2
            metrics["val_oracle_loss"].append(oracle_val_loss)


        metrics["val_loss"].append(val_epoch_loss)
        metrics["val_mi"].append(val_mi)
        metrics["val_r2"].append(r2_val)
        metrics["train_mi"].append(train_mi)
        metrics["train_r2"].append(r2_train)
        

        # ===== EARLY STOPPING =====
        if patience is not None and val_epoch_loss is not None:
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                epochs_without_improvement = 0

                metrics["best_epoch"] = epoch
                metrics["best_val_loss"] = best_val_loss
                state_dict = deepcopy(model.state_dict())
                final_model_path = f"{ckpt_path}{network_name}_s_{seed}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "args": args,
                }, final_model_path)

            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        # ===== LOG =====
        epochs.set_postfix({
            "train": f"{train_epoch_loss:.4f}",
            "val": f"{val_epoch_loss:.4f}" if val_epoch_loss is not None else "NA"
        })

    # ---------- Load best model ----------
    if patience is not None and metrics["best_epoch"] is not None:
        checkpoint = torch.load(final_model_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model, metrics
