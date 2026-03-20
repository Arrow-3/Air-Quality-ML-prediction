import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def training(data, model_choice):

    X = data.drop(columns=['RH', 'Date', 'Time'])
    y = data['RH']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    if model_choice == 'linear':
        model = LinearRegression()

    elif model_choice == 'tree':
        model = DecisionTreeRegressor(min_samples_leaf=3)

    elif model_choice == 'random_forest':
        model = RandomForestRegressor(n_estimators=100)

    else:
        raise ValueError("Invalid model choice")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    return {
        "model": model_choice,
        "y_test": y_test,
        "y_pred": y_pred,
        "rmse": rmse
    }
