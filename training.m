function results = training(data_updated)

    % Define X and y
    X = data_updated{:,[1 2 3 4 5 6 7 8 9 10 12 13 14 15]};
    y = data_updated.RH;

    % Normalize
    X = zscore(X);

    %% Train-Test Split
    split = floor(0.7 * size(X,1));
    X_train = X(1:split,:);
    y_train = y(1:split);

    X_test = X(split+1:end,:);
    y_test = y(split+1:end);

    %% Decision Tree
    tree = fitrtree(X_train, y_train);
    y_pred_tree = predict(tree, X_test);

    rmse_tree = sqrt(mean((y_test - y_pred_tree).^2));

    %% Random Forest (basic version)
    rf = fitrensemble(X_train, y_train);
    y_pred_rf = predict(rf, X_test);

    rmse_rf = sqrt(mean((y_test - y_pred_rf).^2));

    % Store results
    results.y_test = y_test;
    results.y_pred_tree = y_pred_tree;
    results.y_pred_rf = y_pred_rf;
    results.rmse_tree = rmse_tree;
    results.rmse_rf = rmse_rf;

end
