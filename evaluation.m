function evaluation(results)

    disp("Model Used:");
    disp(results.model);

    disp("RMSE:");
    disp(results.rmse);

    % Plot
    figure;
    plot(results.y_test, 'b');
    hold on;
    plot(results.y_pred, 'r');
    legend('Actual','Predicted');

    title(['Model: ', results.model]);
    xlabel('Time Step');
    ylabel('RH');

    saveas(gcf, 'results_plot.png');

end
