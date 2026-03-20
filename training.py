import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def training(data, model_choice):

    X = data.drop(columns=['RH', 'Date', 'Time'])
    y = data['RH']

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Time-aware split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    # ---------------- MODEL SELECTION ---------------- #

    if model_choice == 'linear':
        print("Running Linear Regression...")
        model = LinearRegression()

    elif model_choice == 'tree':
        print("Running Decision Tree...")
        model = DecisionTreeRegressor(min_samples_leaf=3)

    elif model_choice == 'optimized_tree':
        print("Running Optimized Decision Tree (Grid Search)...")

        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_leaf': [1, 3, 5],
            'min_samples_split': [2, 5, 10]
        }

        base_model = DecisionTreeRegressor()

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        print("Best Parameters:", grid_search.best_params_)

    elif model_choice == 'random_forest':
        print("Running Random Forest...")
        model = RandomForestRegressor(n_estimators=100)

    else:
        raise ValueError("Invalid model choice")

    # ---------------- TRAIN ---------------- #
    model.fit(X_train, y_train)

    # ---------------- PREDICT ---------------- #
    y_pred = model.predict(X_test)

    # ---------------- METRICS ---------------- #
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    return {
        "model": model_choice,
        "y_test": y_test,
        "y_pred": y_pred,
        "rmse": rmse
    }
