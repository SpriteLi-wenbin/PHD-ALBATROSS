path = char('..\data\oxford\Path dependent battery degradation dataset\');
list_dataset = ["Group_1", "Group_2", "Group_3", "Group_4"];
num_dataset = length(list_dataset);
cap_n = 1.1;
for idx_dataset = 1:num_dataset
    file_name = dir(path + list_dataset(idx_dataset) + "\*.mat");
    num_file = length(file_name);
    mkdir(path + list_dataset(idx_dataset) + "_csv");  % build folder
    for idx = 1:num_file
            file_path = path + list_dataset(idx_dataset) + "\" + file_name(idx).name;
            %skip_channel_id = skip_channel_list(idx);
            temp_data = load(file_path);  %struct
            name = fieldnames(temp_data);
            temp_data = temp_data.(name{1});
            writetable(temp_data, path + list_dataset(idx_dataset) + "_csv\" + file_name(idx).name(1:end-4) + ".csv");
    end
end

function [] = mydataread(file_name, skip_channel_id)
columns = {'time', 'voltage', 'current', 'capacity_dchg', 'capacity_chrg'};
summary_tab = table();
%folderpath = split(file_name, '/');
%temp_data = split(folderpath(end), '.');
%folderpath = folderpath(1) + "/" + folderpath(2) + "/" + temp_data(1);
folderpath = split(file_name, '.');
folderpath = string(folderpath(1));
temp_data = load(file_name);
date = temp_data.batch_date;
temp_data = temp_data.batch;
num_data = 1;
mkdir(folderpath);
for idx_cell = 1:length(temp_data)
    %% cell sample
    channel_id = temp_data(idx_cell).channel_id;
    if any(strcmp(channel_id, skip_channel_id))
        continue;  %% skip the error testing data
    end
    barcode = temp_data(idx_cell).barcode;
    summary = temp_data(idx_cell).summary;  % summarized information
    cell_data = temp_data(idx_cell).cycles;
    total_cycle = min(length(cell_data), length(summary.QDischarge));
    for idx_cycle = 1:total_cycle
        %% cycle data
        max_cap = summary.QDischarge(idx_cycle);
        R_ohmic = summary.IR(idx_cycle);
        cycle_data = cell_data(idx_cycle);
        if isempty(cycle_data.V)
            continue;  %% skip empty cycle
        end
        temp_tab = table(cycle_data.t, cycle_data.V, cycle_data.I, ...
            cycle_data.Qd, cycle_data.Qc, 'VariableNames', columns);
        tab_size = size(temp_tab);
        temp_tab.num_data = num_data * ones(tab_size(1), 1);
        writetable(temp_tab, folderpath + "/" + barcode + '_cycle_' + string(idx_cycle) + '.csv');  % write cycle data
        temp_tab = table(num_data, idx_cycle, max_cap, barcode, R_ohmic,...
            'VariableNames', {'num_data','step_cycle', 'capacity', 'barcode', 'R_ohmic'});
        %writetable(temp_tab, folderpath + "/" + barcode + '_summary_' + string(idx_cycle) + '.csv');
        summary_tab = [summary_tab; temp_tab];
        num_data = num_data + 1;
    end
end
writetable(summary_tab, folderpath + "/summary.csv");  % write summary info for batch data
end