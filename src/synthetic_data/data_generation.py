import numpy as np
import os

def generate_synthetic_data(num_samples,
                            num_features=1, 
                            function = lambda x: 0.6*x + 0.5,
                            noise_level=0.1,
                            random_seed_x = 42,
                            random_seed_n = 24,):
    # training data
    x = np.random.default_rng(random_seed_x).uniform(-1, 1, (num_samples, num_features))
    true_fcn = function(x)
    sc = true_fcn.std()*noise_level
    noise = np.random.default_rng(random_seed_n).normal(0, sc, true_fcn.shape)
    y = true_fcn + noise  
 

    # reshape outputs
    true_fcn = true_fcn.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return x, y, true_fcn

if __name__ == "__main__":

    def fn_1(x):
        return np.sin(8*x)
    
    def fn_2(x):
        return 0.7*x**6 + 1.2*x**5 -0.15*x**4 -0.4*x**3 -1.1*x**2 -0.5*x +0.1
    def fn_3(x):
        w1 = 5
        b1 = 1
        w2 = 0.15
        b2 = -0.3
        nn = w2*np.tanh(w1*x + b1) + b2
        return np.exp(-5*nn) + 2*nn**2 + 6*np.sin(30*np.abs(nn))**2 + 4.5*x + 3*np.abs(x-0.2)**7.5 + 4
    def fn_4(x):
        a = np.random.default_rng(3).standard_normal(size=(20, 1))
        return x@a
    
    functions = [(fn_1, 1, "sin"), (fn_2, 1, "polynomial"), (fn_3, 1, "tanh"), (fn_4, 20, "linear")]
    noise_levels = {"easy":0.05, "hard":0.1}
    for lev in noise_levels:
        for func, dim, name in functions:
            x, y, true_fcn = generate_synthetic_data(
            num_samples=100_000,
            num_features=dim,
            function=func,
            noise_level=noise_levels[lev]
            )       
            root_dir = f"./array_data/{name}_{lev}"
            os.makedirs(root_dir, exist_ok=True)
            name = f"{name}_{lev}.npz"
            final_path = os.path.join(root_dir, name)
            np.savez_compressed(final_path,x = x, y=y,oracle=true_fcn)