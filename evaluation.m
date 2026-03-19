function evaluation(results)

    disp("Decision Tree RMSE:");
    disp(results.rmse_tree);

    disp("Random Forest RMSE:");
    disp(results.rmse_rf);

    % Plot
    figure;
    plot(results.y_test, 'b');
    hold on;
    plot(results.y_pred_rf, 'r');
    legend('Actual','Predicted');
    title('Random Forest Prediction');
    xlabel('Time Step');
    ylabel('RH');

    saveas(gcf, 'results_plot.png');

end
