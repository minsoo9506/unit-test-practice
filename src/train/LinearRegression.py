from sklearn.linear_model import LinearRegression

def train_LinearRegression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model