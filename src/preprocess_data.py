import os
import numpy as np
import pandas as pd
from metrics.tsp import tsp
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
from scipy.io import loadmat
from sklearn.datasets import fetch_california_housing

def gaussianize_features(X):
    qt = QuantileTransformer(output_distribution='normal', random_state=0)
    X_gauss = qt.fit_transform(X)
    return X_gauss

def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def minmax_scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def delete_missing_rows(X, Y):
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
    return X[mask], Y[mask]

if __name__ == "__main__":
    # SARCOS Inverse Dynamics dataset
    print("Preprocessing SARCOS Inverse Dynamics dataset...")
    sarcos_path = "./raw_data/sarcos/"
    sarcos_preprocessed_path = "./array_data/sarcos/"

    mat_data = loadmat(f"{sarcos_path}sarcos_inv.mat")
    mat_data_test = loadmat(f"{sarcos_path}sarcos_inv_test.mat")

    data = mat_data.get('sarcos_inv', None)
    data_test = mat_data_test.get('sarcos_inv_test', None)
    data = data.astype(np.float32)
    data_test = data_test.astype(np.float32)

    X_train = data[:, 0:21]
    Y_train = data[:, 21].reshape(-1, 1)
    X_test = data_test[:, 0:21]
    Y_test = data_test[:, 21].reshape(-1, 1)
    np.savez_compressed(f"{sarcos_preprocessed_path}sarcos_inv.npz", x=X_train, y=Y_train)
    np.savez_compressed(f"{sarcos_preprocessed_path}sarcos_inv_test.npz", x=X_test, y=Y_test)
    print(f"Preprocessed SARcos Inverse Dynamics data saved to {sarcos_preprocessed_path}sarcos_inv.npz")

    print("Preprocessing Housing dataset...")
    #housing_path = "./raw_data/housing/"
    #housing_preprocessed_path = "./array_data/housing/"

    #df = pd.read_csv(f"{housing_path}housing.csv")
    X, Y = fetch_california_housing(data_home="./raw_data/housing", download_if_missing=True,return_X_y=True)
    housing_preprocessed_path = "./array_data/"
    print(f"housing X shape :{X.shape}")
    X = X.astype(np.float32)
    Y = Y.astype(np.float32).reshape(-1, 1)
    np.savez_compressed(f"{housing_preprocessed_path}housing.npz", x=X, y=Y)
    print(f"Preprocessed Housing data saved to {housing_preprocessed_path}housing.npz")
    
    mutual_info_scores = [tsp(X[:, k].reshape(-1,1), Y) for k in range(X.shape[1])]
    indices = np.argsort(mutual_info_scores)[::-1]
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. feature {indices[f]} ({mutual_info_scores[indices[f]]})") 
    selected_features = indices[:2]
    X_selected = X[:, selected_features]
    print(f"selected features num: {X_selected.shape[1]}")
    housing_reduced_path = "./array_data/reduced_housing/"
    os.makedirs(housing_reduced_path, exist_ok=True)
    np.savez_compressed(f"{housing_reduced_path}reduced_housing.npz", x=X_selected, y=Y)

    X_unselected = X[:, indices[3:]]
    X_unselected = X_unselected[:, -2:]
    housing_unfavorable_path = "./array_data/unfavorable_housing/"
    os.makedirs(housing_unfavorable_path, exist_ok=True)
    np.savez_compressed(f"{housing_unfavorable_path}unfavorable_housing.npz", x=X_selected, y=Y)

    X_selected_2 = X[:, indices[:3]]
    housing_reduced_2_path = "./array_data/reduced_housing_2/"
    os.makedirs(housing_reduced_2_path, exist_ok=True)
    np.savez_compressed(f"{housing_reduced_2_path}reduced_housing_2.npz", x=X_selected_2, y=Y)
    print(f"Preprocessed Reduced Housing data 2 saved to {housing_reduced_2_path}reduced_housing_2.npz")

    housing_unfavorable_path = "./array_data/unfavorable_housing_2/"
    os.makedirs(housing_unfavorable_path, exist_ok=True)
    np.savez_compressed(f"{housing_unfavorable_path}unfavorable_housing_2.npz", x=X_selected_2, y=Y)
    print("Housing Dataset complete")
    housing_full_path = "./array_data/housing/"
    os.makedirs(housing_full_path, exist_ok=True)
    np.savez_compressed(f"{housing_full_path}housing.npz", x=X, y=Y)

    unfavorable_housing_full_path = "./array_data/unfavorable_housing_full/"
    os.makedirs(unfavorable_housing_full_path, exist_ok=True)
    np.savez_compressed(f"{unfavorable_housing_full_path}unfavorable_housing_full.npz", x=X, y=Y)
    
    print("Preprocessing Combined Cycle Power Plant dataset...")
    ccpp_path = "./raw_data/ccpp/"
    ccpp_preprocessed_path = "./array_data/ccpp/"
    df = pd.read_excel(f"{ccpp_path}ccpp_data.xlsx")
    print("Original data shape:", df.shape)
    data = df.to_numpy().astype(np.float32)
    X = data[:, :4]
    Y = data[:, 4].reshape(-1, 1)
    np.savez_compressed(f"{ccpp_preprocessed_path}ccpp.npz", x=X, y=Y)
    print(f"Preprocessed CCPP data saved to {ccpp_preprocessed_path}ccpp.npz")
    mutual_info_scores_ccpp = [tsp(X[:, k].reshape(-1,1), Y) for k in range(X.shape[1])]
    indices_ccpp = np.argsort(mutual_info_scores_ccpp)[::-1]
    print("CCPP Feature ranking:")
    for f in range(X.shape[1]): 
        print(f"{f + 1}. feature {indices_ccpp[f]} ({mutual_info_scores_ccpp[indices_ccpp[f]]})")
    print("Preprocessing Noisy CCPP dataset...")
    X_unselected = X[:, indices_ccpp[-2:]]
    noisy_ccpp_path = "./array_data/noisy_ccpp/"
    os.makedirs(noisy_ccpp_path, exist_ok= True)

    np.savez_compressed(f"{noisy_ccpp_path}noisy_ccpp.npz", x=X, y=Y)    
