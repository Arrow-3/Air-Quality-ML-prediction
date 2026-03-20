function results = training(data_updated, model_choice)

    % Set seed for reproducibility
    rng(42);

    % Define X and y
    X = data_updated{:,[1 2 3 4 5 6 7 8 9 10 12 13 14 15]};
    y = data_updated.RH;

    % Normalize
    X = zscore(X);

    % Train-Test Split (time-aware)
    split = floor(0.7 * size(X,1));
    X_train = X(1:split,:);
    y_train = y(1:split);

    X_test = X(split+1:end,:);
    y_test = y(split+1:end);

    switch model_choice

        %% ---------------- LINEAR REGRESSION ----------------
        case 'linear'
            disp("Running Linear Regression...");
            mdl = fitlm(X_train, y_train);
            y_pred = predict(mdl, X_test);

        %% ---------------- DECISION TREE ----------------
        case 'tree'
            disp("Running Decision Tree...");
            mdl = fitrtree(X_train, y_train, ...
                'MinLeafSize',3, ...
                'MinParentSize',1);
            y_pred = predict(mdl, X_test);

        %% ---------------- OPTIMIZED TREE (GRID SEARCH) ----------------
        case 'optimized_tree'
            disp("Running Optimized Decision Tree (Grid Search)...");

            % Hyperparameters
            maxSplits = optimizableVariable('MaxNumSplits',[1,50],'Type','integer');
            minLeaf = optimizableVariable('MinLeafSize',[1,10],'Type','integer');

            opts = struct(...
                'Optimizer','gridsearch', ...
                'MaxObjectiveEvaluations',50, ...
                'CVPartition',cvpartition(size(X_train,1),'KFold',3), ...
                'Verbose',1);

            mdl = fitrtree(X_train, y_train, ...
                'OptimizeHyperparameters',{'MaxNumSplits','MinLeafSize'}, ...
                'HyperparameterOptimizationOptions',opts);

            y_pred = predict(mdl, X_test);

        %% ---------------- RANDOM FOREST (BAYESIAN OPTIMIZATION) ----------------
        case 'random_forest'
            disp("Running Random Forest (Bayesian Optimization)...");

            % Define hyperparameters
            params = [
                optimizableVariable('NumLearningCycles',[10,250],'Type','integer')
                optimizableVariable('MaxNumSplits',[1,50],'Type','integer')
                optimizableVariable('MinLeafSize',[1,25],'Type','integer')
                optimizableVariable('MinParentSize',[2,50],'Type','integer')
            ];

            % Objective function
            objectiveFcn = @(params) rfObjective(params, X_train, y_train);

            resultsBO = bayesopt(objectiveFcn, params, ...
                'MaxObjectiveEvaluations',15, ...
                'Verbose',1, ...
                'AcquisitionFunctionName','expected-improvement-plus');

            bestParams = resultsBO.XAtMinObjective;

            disp("Best Parameters:");
            disp(bestParams);

            % Train final model
            treeTemplate = templateTree( ...
                'MaxNumSplits', bestParams.MaxNumSplits, ...
                'MinLeafSize', bestParams.MinLeafSize, ...
                'MinParentSize', bestParams.MinParentSize);

            mdl = fitrensemble(X_train, y_train, ...
                'Method','LSBoost', ...
                'NumLearningCycles', bestParams.NumLearningCycles, ...
                'Learners', treeTemplate);

            y_pred = predict(mdl, X_test);

        otherwise
            error("Invalid model choice");
    end

    %% ---------------- METRICS ----------------
    rmse = sqrt(mean((y_test - y_pred).^2));

    %% ---------------- STORE RESULTS ----------------
    results.y_test = y_test;
    results.y_pred = y_pred;
    results.rmse = rmse;
    results.model = model_choice;

end


%% ================= OBJECTIVE FUNCTION =================
function mse = rfObjective(params, X_train, y_train)

    treeTemplate = templateTree( ...
        'MaxNumSplits', params.MaxNumSplits, ...
        'MinLeafSize', params.MinLeafSize, ...
        'MinParentSize', params.MinParentSize);

    mdl = fitrensemble(X_train, y_train, ...
        'Method','LSBoost', ...
        'NumLearningCycles', params.NumLearningCycles, ...
        'Learners', treeTemplate);

    cvModel = crossval(mdl, 'KFold', 3);
    mse = kfoldLoss(cvModel);

end
