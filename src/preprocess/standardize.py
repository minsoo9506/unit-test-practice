import numpy as np
from sklearn.preprocessing import StandardScaler

def standardize(X):
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(X)
     return X_scaled