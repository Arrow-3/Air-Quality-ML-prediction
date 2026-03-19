function data_updated = preprocessing(filepath)

    data = readtable(filepath);

    % Fix decimal format
    A = {'CO_GT_', 'C6H6_GT_', 'T', 'RH', 'AH'};
    for i = 1:length(A)
        colName = A{i};
        if iscell(data.(colName))
            data.(colName) = strrep(data.(colName), ',', '.');
        end
        data.(colName) = str2double(data.(colName));
    end

    % Time formatting
    data.Time = strrep(data.Time, '.', ':');

    data.Hour = hour(data.Time);
    data.Month = month(data.Date);
    data.DayOfWeek = weekday(data.Date);

    % Remove bad column
    data = removevars(data, 'NMHC_GT_');

    % Replace -200 with hourly mean
    data_mod = data(:, 3:14);

    for col = 1:width(data_mod)
        colName = data_mod.Properties.VariableNames{col};
        for h = 0:23
            rows = data.Hour == h;
            values = data_mod{rows, colName};

            valid = values(values ~= -200);
            if ~isempty(valid)
                meanVal = mean(valid);
                data_mod{rows & data_mod{:, colName} == -200, colName} = meanVal;
            end
        end
    end

    % Add time features
    data_updated = data_mod;
    data_updated.Hour = data.Hour;
    data_updated.Month = data.Month;
    data_updated.DayOfWeek = data.DayOfWeek;

end
