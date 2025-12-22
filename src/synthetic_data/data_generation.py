import numpy as np


def generate_synthetic_data(num_samples_train,
                            num_samples_test, 
                            num_features=1, 
                            function = lambda x: 0.6*x + 0.5,
                            noise_level=0.1,
                            random_seed_x = 42,
                            random_seed_n = 24,
                            random_seed_x_test = 11,
                            random_seed_n_test = 7):
    # training data
    x_train = np.random.default_rng(random_seed_x).uniform(-1, 1, (num_samples_train, num_features))
    true_fcn_train = function(x_train)
    noise_train = np.random.default_rng(random_seed_n).normal(0, noise_level, true_fcn_train.shape)
    y_train = true_fcn_train + noise_train    
    
    # test data
    x_test = np.random.default_rng(random_seed_x_test).uniform(-1, 1, (num_samples_test, num_features))
    true_fcn_test = function(x_test)
    noise_test = np.random.default_rng(random_seed_n_test).normal(0, noise_level, true_fcn_test.shape)
    y_test = true_fcn_test + noise_test

    # reshape outputs
    true_fcn_train = true_fcn_train.reshape(-1, 1)
    true_fcn_test = true_fcn_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return x_train, y_train, true_fcn_train, x_test, y_test, true_fcn_test

if __name__ == "__main__":
    x_train, y_train, true_fcn_train, x_test, y_test, true_fcn_test = generate_synthetic_data(
        num_samples_train=1000,
        num_samples_test=200,
        num_features=1,
        function=lambda x: np.sin(3 * x) + 0.4,
        noise_level=0.1
    )
    print("Training data shapes:", x_train.shape, y_train.shape)
    print("Test data shapes:", x_test.shape, y_test.shape)