clc;
clear;
close all;

disp("=== Air Quality Prediction Pipeline Started ===");


filepath = 'C:\Drive W\Machine Learning\AirQualityUCI.csv';
data = readtable(filepath);
disp(data);
summary(data)

A = {'CO_GT_', 'C6H6_GT_', 'T', 'RH', 'AH'};  % Column names in MATLAB syntax

% Loop through each column in A
for i = 1:length(A)
    colName = A{i};  % Get the column name

    % Replace commas with dots in the selected column (if it's a cell array of strings)
    if iscell(data.(colName))
        data.(colName) = strrep(data.(colName), ',', '.');
    end
    
    % Convert the column to numeric (if it's still a string)
    data.(colName) = str2double(data.(colName));
end

% Display a few rows to check the conversion
head(data)

data_num = data(:,3:end);
% Checking for zeros in the data
zeroValues = sum(data_num{:,:} == -200, 1);  % Counts zero values per column
disp('Zero values per column:')
disp(zeroValues) 

% Replace periods with colons in the 'Time' column
data.Time = strrep(data.Time, '.', ':');

% Extract the hour from the 'Time' column and create a new 'Hour' column
data.Hour = hour(data.Time);

% Extract the month from the 'Date' column and create a new 'Month' column
data.Month = month(data.Date);

% Extract 'Day of the Week' from 'Date' (1=Sunday, 7=Saturday)
data.DayOfWeek = weekday(data.Date);

% Display the first few rows to check the new 'Month' column
head(data);

% Remove the 'NMHC_GT' column
data = removevars(data, 'NMHC_GT_');

% Display the first few rows to check if the column is removed
head(data);

data_num = data(:,3:end);
% Checking for zeros in the data
zeroValues = sum(data_num{:,:} == -200, 1);  % Counts zero values per column
disp('Missing values per column:')
disp(zeroValues) 

% Extract numeric data (excluding the first two columns for Date and Time)
data_mod = data(:, 3:14);

% Replace -200 values with the hourly mean
for col = 1:width(data_mod)
    % Get the current column
    colName = data_mod.Properties.VariableNames{col};
    
    % Loop over each hour
    for h = 0:23
        % Identify rows for the current hour
        rowsForHour = data.Hour == h;
        
        % Get the values for this hour in the current column
        valuesForHour = data_mod{rowsForHour, colName};
        
        % Calculate the mean for non -200 values
        validValues = valuesForHour(valuesForHour ~= -200);
        if ~isempty(validValues)
            meanForHour = mean(validValues);
            
            % Replace -200 values with the hourly mean
            data_mod{rowsForHour & data_mod{:, colName} == -200, colName} = meanForHour;
        end
    end
end

% Display the modified dataset
head(data_mod);

% Checking for zeros in the data
zeroValues = sum(data_mod{:,:} == -200, 1);  % Counts zero values per column
disp('Missing values per column:')
disp(zeroValues) 

% Add the 'Hour' and 'Month' columns to the 'data_mod' table
data_updated = addvars(data_mod, data.Hour, 'After', width(data_mod), 'NewVariableNames', 'Hour');
data_updated = addvars(data_updated, data.Month, 'After', width(data_updated), 'NewVariableNames', 'Month');
data_updated = addvars(data_updated, data.DayOfWeek, 'After', width(data_updated), 'NewVariableNames', 'DayOfWeek');

% Display the first few rows to verify the concatenation
head(data_updated);


% Calculate the correlation matrix for numeric data 
correlationMatrix = corr(table2array(data_updated), 'Rows', 'complete');

% Generate a heatmap
h = heatmap(data_updated.Properties.VariableNames, data_updated.Properties.VariableNames, correlationMatrix);

% Customize the heatmap
h.Colormap = parula;   % Use the 'viridis' colormap
h.Title = 'Heatmap of Correlation between Variables';
h.FontSize = 12;

% Display the heatmap
h.ColorLimits = [-1, 1];  % Set color limits to emphasize correlation values between -1 and 1

% Get the list of all features 
features = data_updated.Properties.VariableNames; 

% Create a figure
figure;

% Loop through each feature and plot it against 'RH'
for i = 1:length(features)
    subplot(ceil(length(features)/3), 3, i);  % Create a subplot grid
    featureName = features{i};  % Get the feature name
    
    % Scatter plot of the feature against RH
    scatter(data.(featureName), data.RH, '.');
    hold on;
    
    % Fit a linear model to the feature and RH
    lm = fitlm(data.(featureName), data.RH);
    
    % Plot the linear fit
    plot(lm);
    
    % Customize plot
    title(['RH vs ', featureName]);
    xlabel(featureName);
    ylabel('RH');
    legend off;  % Turn off the legend to avoid duplication
    hold off;
end

% Adjust the figure layout
sgtitle('Scatter Plots of Features vs RH with Linear Fits');



% Define Feature (X) and Target (y)
% X = All columns except 'RH' (excluding Date and Time)
X = data_updated{:,[1 2 3 4 5 6 7 8 9 10 12 13 14 15]};  % X-input features (ignoring first 2 columns for Date and Time, last 3 including RH)
y = data_updated.RH;           % y-output feature (RH)

% Normalize Feature variables using zscore (standardization)
X_std = zscore(X);


%% Linear regression

%Since the pollutants are having different frequencies varying over time and month during entire year,
% it should be necessary to include date and time columns in features

%When the variables in a dataset vary randomly over time without clear temporal dependencies,
% a standard regression model may be more appropriate, but the inclusion of time-related features
% (like hour, day, month) could still be beneficial for capturing potential time-based patterns or influences.

%In your case, the pollutants or environmental conditions (like temperature, humidity, etc.) 
% may exhibit variations tied to time of day, seasonal changes, or other cyclical patterns,
% but they may not necessarily follow a strict periodic time series pattern.

% Define number of folds for cross-validation
k = 10;

% Initialize partition for k-fold cross-validation
cv = cvpartition(size(X_std, 1), 'KFold', k);

% Initialize vectors to store errors, intercepts, and coefficients for each fold
mse_values = zeros(k, 1);         % Mean Squared Error for each fold
rmse_values = zeros(k, 1);        % Root Mean Squared Error for each fold
intercepts = zeros(k, 1);         % Store intercepts for each fold
coefficients = zeros(k, size(X_std, 2));  % Store coefficients for each fold (features)

for i = 1:k
    % Training and testing sets for the current fold
    X_train = X_std(training(cv, i), :);  % Training data
    y_train = y(training(cv, i), :);      % Training labels
    
    X_test = X_std(test(cv, i), :);       % Testing data
    y_test = y(test(cv, i), :);           % Testing labels
    
    % Train a linear regression model on the current fold
    lm = fitlm(X_train, y_train);

    % Predict using the model on the training set (to calculate training loss)
    y_train_pred = predict(lm, X_train);

    % Predict using the model on the test set
    y_pred = predict(lm, X_test);
    
    % Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse_train_values(i) = mean((y_train - y_train_pred).^2);  % Training MSE
    rmse_train_values(i) = sqrt(mse_train_values(i));        % RMSE

    mse_values(i) = mean((y_test - y_pred).^2);  % MSE
    mae_values(i) = mean(abs(y_test - y_pred));  % MAE
    rmse_values(i) = sqrt(mse_values(i));        % RMSE
    
    % Store intercept and coefficients for each fold
    intercepts(i) = lm.Coefficients.Estimate(1);     % Store intercept
    coefficients(i, :) = lm.Coefficients.Estimate(2:end)';  % Store coefficients (slopes)
end

% Compute the average MSE and RMSE across all folds
rmse_train = mean(rmse_train_values);

mean_mse = mean(mse_values);
mean_mae = mean(mae_values);
mean_rmse = mean(rmse_values);

% Compute the average intercept and coefficients across all folds
average_intercept = mean(intercepts);
average_coefficients = mean(coefficients, 1);  % Average across folds

% Display the cross-validated RMSE and MSE
disp('Cross-validated Mean Squared Error (MSE):');
disp(mean_mse);
disp('Cross-validated Mean Absolute Error (MAE):');
disp(mean_mae);
disp('Cross-validated Root Mean Squared Error (RMSE):');
disp(mean_rmse);

% Display the average intercept and coefficients
disp('Average Intercept:');
disp(average_intercept);
disp('--------------------------------');
disp('Average Coefficients (Slope):');
disp(average_coefficients);

% Display the RMSE value
disp('Training RMSE of model:');
disp(rmse_train);

disp('Baseline RMSE of model:');
disp(mean_rmse);

% Display 10 actual vs predicted data points
disp('First 10 actual vs predicted values:');
disp(table(y_test(1:10), y_pred(1:10), 'VariableNames', {'Actual', 'Predicted'}));

%}



% Step 2: Train-Test Split (use 70% of data for training, 30% for testing)
split_ratio = 0.7;
split_index = floor(split_ratio * size(X, 1));

X_train = X(1:split_index, :);   % Training features
y_train = y(1:split_index, :);   % Training target

X_test = X(split_index+1:end, :);  % Testing features
y_test = y(split_index+1:end, :);  % Testing target


%% Decision Tree regression [C]

% Step 3: Train Decision Tree Regression Model
tree_model = fitrtree(X_train, y_train, 'MinParentSize',1, 'MinLeafSize',3, 'PredictorSelection','curvature', 'Surrogate','on');  % Train decision tree on training data

% Predict using the model on the training set (to calculate training loss)
y_train_pred = predict(tree_model, X_train);

rmse_train = sqrt(mean((y_train - y_train_pred).^2));  % Calculate Root Mean Squared Error (RMSE)
disp('Training loss RMSE of Decision Tree Regression Model:');
disp(rmse_train);

% Step 4: Predict on Test Data
y_pred = predict(tree_model, X_test);  % Make predictions on test data

% Step 5: Evaluate Model Performance
rmse = sqrt(mean((y_test - y_pred).^2));  % Calculate Root Mean Squared Error (RMSE)
disp('RMSE of Decision Tree Regression Model:');
disp(rmse);

% Step 6: Plot Predictions vs Actual Values
figure;
plot(y_test, 'b', 'DisplayName', 'Actual');
hold on;
plot(y_pred, 'r--', 'DisplayName', 'Predicted');
xlabel('Time Step');
ylabel('Relative Humidity (RH)');
title('Decision Tree Regression: Actual vs Predicted');
legend;
hold off;

% Step 7: Visualize the Tree
view(tree_model, 'Mode', 'graph');  % Visualize the decision tree

% Display 10 actual vs predicted data points
disp('First 10 actual vs predicted values:');
disp(table(y_test(1:10), y_pred(1:10), 'VariableNames', {'Actual', 'Predicted'}));

%}


%% Gridsearch Decision tree [C]

% Define Hyperparameter Ranges for Grid Search
maxNumSplits = optimizableVariable('MaxNumSplits', [1, 50], 'Type', 'integer');
minLeafSize = optimizableVariable('MinLeafSize', [1, 10], 'Type', 'integer');
numVariablesToSample = optimizableVariable('NumVariablesToSample', [1, size(X_train, 2)], 'Type', 'integer');

% Set up Hyperparameter Optimization Options
opts = struct(...
    'Optimizer', 'gridsearch', ...                      % Set optimizer to grid search
    'MaxObjectiveEvaluations', 50, ...                  % Limit evaluations to 50
    'CVPartition', cvpartition(size(X_train, 1), 'KFold', 3), ... % 3-Fold cross-validation
    'Verbose', 1 ...                                    % Set verbosity for display
    );

% Train the Decision Tree Model using Grid Search
tuning_model = fitrtree(...
    X_train, y_train, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', opts);


% Display the structure of the HyperparameterOptimizationResults
disp(tuning_model.HyperparameterOptimizationResults);

% Predict using the model on the training set (to calculate training loss)
y_train_pred = predict(tuning_model, X_train);

rmse_train = sqrt(mean((y_train - y_train_pred).^2));  % Calculate Root Mean Squared Error (RMSE)
disp('Training loss RMSE of Decision Tree Regression Model with GridSearch:');
disp(rmse_train);

% Predict on Test Data
y_pred = predict(tuning_model, X_test);  % Predictions on the test data

% Calculate RMSE (Root Mean Squared Error)
rmse = sqrt(mean((y_test - y_pred).^2));
disp(['RMSE of Optimized Decision Tree Regression Model : ', num2str(rmse)]);

% Plot Predictions vs Actual Values
figure;
plot(y_test, 'b', 'DisplayName', 'Actual');
hold on;
plot(y_pred, 'r--', 'DisplayName', 'Predicted');
xlabel('Time Step');
ylabel('Relative Humidity (RH)');
title('Decision Tree Regression: Actual vs Predicted');
legend;
hold off;
%}




%% Random Forest regression   [C]
% Train a Random Forest model
%% Bayesian Optimizer for Random Forest

% Set random seed for reproducibility
%rng(20); 

% Split data into training and testing sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% train, 20% test
X_train = X(training(cv), :);
X_test = X(test(cv), :);
y_train = y(training(cv), :);
y_test = y(test(cv), :);

% Define the hyperparameters to tune
params = [
    optimizableVariable('n_estimators', [10, 250], 'Type', 'integer');     % Number of trees
    optimizableVariable('MaxNumSplits', [1, 50], 'Type', 'integer');       % Maximum number of splits
    optimizableVariable('MinLeafSize', [1, 25], 'Type', 'integer');        % Minimum number of leaf nodes
    optimizableVariable('MinParentSize', [2, 50], 'Type', 'integer')       % Minimum parent size
];

% Define the objective function for Random Forest
% This function trains a Random Forest and returns the negative MSE for Bayesian optimization
objectiveFcn = @(params) fitAndEvaluateRF(params, X_train, y_train);

function mse = fitAndEvaluateRF(params, X_train, y_train)
    % Use a templateTree for setting tree-specific parameters
    treeTemplate = templateTree('MaxNumSplits', params.MaxNumSplits, ...
                                'MinLeafSize', params.MinLeafSize, ...
                                'MinParentSize', params.MinParentSize);

    % Train Random Forest model using fitrensemble
    mdl = fitrensemble(X_train, y_train, ...
        'Method', 'LSBoost', ...                              % Bagging method for Random Forest
        'NumLearningCycles', params.n_estimators, ...     % Number of trees
        'Learners', treeTemplate);                       % Use templateTree for tree parameters
        
    
    % Perform cross-validation to compute mean squared error (MSE)
    cvModel = crossval(mdl, 'KFold', 3);
    mse = kfoldLoss(cvModel, 'LossFun', 'mse');
    
    % Return negative MSE since Bayesian optimization minimizes the objective
    mse = -mse;

end

% Perform Bayesian Optimization
results = bayesopt(objectiveFcn, params, ...
    'MaxObjectiveEvaluations', 15, ...    % Number of iterations for optimization
    'Verbose', 1, ...
    'IsObjectiveDeterministic', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus'); % Acquisition function for optimization

% Extract the best parameters
bestParams = results.XAtMinObjective;

% Display the best hyperparameters
disp('Best Hyperparameters found:');
disp(bestParams);

% Train the final Random Forest model with the best hyperparameters
treeTemplateFinal = templateTree('MaxNumSplits', bestParams.MaxNumSplits, ...
                                 'MinLeafSize', bestParams.MinLeafSize, ...
                                 'MinParentSize', bestParams.MinParentSize);

bestModel_RF = fitrensemble(X_train, y_train, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', bestParams.n_estimators, ...
    'Learners', treeTemplateFinal); % Final model with best parameters

% Predict using the model on the training set (to calculate training loss)
y_train_pred = predict(bestModel_RF, X_train);

rmse_train = sqrt(mean((y_train - y_train_pred).^2));  % Calculate Root Mean Squared Error (RMSE)
disp('Training loss RMSE of Random Forest Regression Model:');
disp(rmse_train);

% Evaluate the model on the test set
y_pred = predict(bestModel_RF, X_test);

% Compute MSE and RMSE on the test set
mse = mean((y_test - y_pred).^2);
rmse = sqrt(mse);

% Display the results
disp('Mean Squared Error (MSE) on Test Set of Random Forest Regression Model:');
disp(mse);
disp('Root Mean Squared Error (RMSE) on Test Set of Random Forest Regression Model:');
disp(rmse);

% Display 10 actual vs predicted data points
disp('First 10 actual vs predicted values:');
disp(table(y_test(1:10), y_pred(1:10), 'VariableNames', {'Actual', 'Predicted'}));

% Plot Predictions vs Actual Values
figure;
plot(y_test, 'b', 'DisplayName', 'Actual');
hold on;
plot(y_pred, 'r', 'DisplayName', 'Predicted');
xlabel('Time Step');
ylabel('Relative Humidity (RH)');
title('Random Forest Regression: Actual vs Predicted');
legend;
hold off;
%}


