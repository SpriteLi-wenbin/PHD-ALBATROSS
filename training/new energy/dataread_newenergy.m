file_name = {'2017-05-12_batchdata_updated_struct_errorcorrect.mat', '2017-06-30_batchdata_updated_struct_errorcorrect.mat', '2018-04-12_batchdata_updated_struct_errorcorrect.mat'};
skip_channel_list = {{'13', '19', '21', '22', '31'},{'1', '2', '3', '5', '6'},{'46'}};
file_name = {'2018-04-12_batchdata_updated_struct_errorcorrect.mat'};
skip_channel_list = {{'46'}};
cap_n = 1.1;
num_file = length(file_name);
for idx = 1:num_file
    file = file_name{idx};
    skip_channel_id = skip_channel_list(idx);
    mydataread(file, skip_channel_id);
end

function [] = mydataread(file_name, skip_channel_id)
columns = {'time', 'voltage', 'current', 'capacity_dchg', 'capacity_chrg', 'Temperature'};
%folderpath = split(file_name, '/');
%temp_data = split(folderpath(end), '.');
%folderpath = folderpath(1) + "/" + folderpath(2) + "/" + temp_data(1);
folderpath = split(file_name, '.');
folderpath = folderpath{1};
temp_data = load(file_name);
date = temp_data.batch_date;
temp_data = temp_data.batch;
num_data = 1;
mkdir(sprintf('%s/summary',folderpath));
mkdir(sprintf('%s/data',folderpath));
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
            cycle_data.Qd, cycle_data.Qc, cycle_data.T, 'VariableNames', columns);
        tab_size = height(temp_tab);
        temp_tab.num_data = num_data * ones(tab_size, 1);
        filepath = sprintf('%s/data/%s_cycle_%s.csv', folderpath, barcode, string(idx_cycle));
        writetable(temp_tab, filepath);  % write cycle data
        %temp_tab = table(num_data, idx_cycle, max_cap, barcode, R_ohmic,...
        %    'VariableNames', {'num_data','step_cycle', 'capacity', 'barcode', 'R_ohmic'});
        %writetable(temp_tab, folderpath + "/" + barcode + '_summary_' + string(idx_cycle) + '.csv');
        num_data = num_data + 1;
    end
    summary_tab = struct2table(summary);
    filepath = sprintf('%s/summary/%s_summary.csv', folderpath, barcode);
    writetable(summary_tab, filepath);  % write summary info for batch data
end
end