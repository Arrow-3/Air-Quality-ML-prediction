function results = training(data_updated, model_choice)

    % Define X and y
    X = data_updated{:,[1 2 3 4 5 6 7 8 9 10 12 13 14 15]};
    y = data_updated.RH;

    % Normalize
    X = zscore(X);

    % Train-Test Split
    split = floor(0.7 * size(X,1));
    X_train = X(1:split,:);
    y_train = y(1:split);

    X_test = X(split+1:end,:);
    y_test = y(split+1:end);

    switch model_choice

        case 'linear'
            disp("Running Linear Regression...");
            mdl = fitlm(X_train, y_train);
            y_pred = predict(mdl, X_test);

        case 'tree'
            disp("Running Decision Tree...");
            mdl = fitrtree(X_train, y_train);
            y_pred = predict(mdl, X_test);

        case 'optimized_tree'
            disp("Running Optimized Decision Tree...");
            mdl = fitrtree(X_train, y_train, ...
                'MinLeafSize',3, 'MinParentSize',1);
            y_pred = predict(mdl, X_test);

        case 'random_forest'
            disp("Running Random Forest...");
            mdl = fitrensemble(X_train, y_train);
            y_pred = predict(mdl, X_test);

        otherwise
            error("Invalid model choice");
    end

    % Compute RMSE
    rmse = sqrt(mean((y_test - y_pred).^2));

    % Store results
    results.y_test = y_test;
    results.y_pred = y_pred;
    results.rmse = rmse;
    results.model = model_choice;

end
