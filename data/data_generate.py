from sklearn.datasets import make_regression

def generate_data(n_samples=100, n_features=10, bias=1):
    X, y, w = make_regression(n_samples=n_samples, n_features=n_features, coef=True,
                            random_state=0, bias=bias)
    return X, y, w