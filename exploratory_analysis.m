function exploratory_analysis(data, data_updated)

    disp("Running Exploratory Data Analysis...");

    %% ---------------- CORRELATION HEATMAP ----------------
    correlationMatrix = corr(table2array(data_updated), 'Rows', 'complete');

    figure;
    h = heatmap(data_updated.Properties.VariableNames, ...
                data_updated.Properties.VariableNames, ...
                correlationMatrix);

    h.Colormap = parula;
    h.Title = 'Correlation Heatmap';
    h.FontSize = 12;
    h.ColorLimits = [-1, 1];

    %% ---------------- SCATTER PLOTS ----------------
    features = data_updated.Properties.VariableNames;

    figure;

    for i = 1:length(features)

        featureName = features{i};

        subplot(ceil(length(features)/3), 3, i);

        scatter(data_updated.(featureName), data_updated.RH, '.');
        hold on;

        % Fit linear model
        lm = fitlm(data_updated.(featureName), data_updated.RH);
        plot(lm);

        title(['RH vs ', featureName]);
        xlabel(featureName);
        ylabel('RH');

        legend off;
        hold off;
    end

    sgtitle('Feature vs RH Relationships');

end
