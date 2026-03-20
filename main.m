clc;
clear;
close all;

disp("=== Air Quality Prediction Pipeline Started ===");

filepath = 'data/AirQualityUCI.csv';

% Step 1: Preprocessing
[data, data_updated] = preprocessing(filepath);

% Step 2: EDA
exploratory_analysis(data, data_updated);

% Step 3: Model Selection
model_choice = 'random_forest';

% Step 4: Training
results = training(data_updated, model_choice);

% Step 5: Evaluation
evaluation(results);
