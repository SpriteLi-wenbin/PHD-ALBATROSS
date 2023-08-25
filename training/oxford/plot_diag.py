import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

colors = np.array([(0, 0, 0), (0, 84, 159), (122, 181, 29), (242, 148, 0), (218, 31, 61), (48, 102, 109),
                   (199, 221, 242), (230, 247, 203)]) / 255
markers = np.array(['o', 's', '^', 'X', '*', 'h', 'D', '+'])
fig_size = (4, 3)
legend_size = 14
resolution = 300
axis_font = 16
tick_size = 14

cap_n = 3.0
eps = 1.0
soh_label = np.array([100, 95, 90, 85, 80])

fig = plt.figure(figsize=fig_size, dpi=resolution)
ax = plt.axes()

# plot capacity sequence
status = 'chrg'
path = r'D:\BMS\data\oxford\Path dependent battery degradation dataset\Group_1_csv\capInLoop'
list_file = os.listdir('{0}\{1}'.format(path, status))
cnt = 0
ax.clear()
list_legend_label = []
list_legend_idx = []
for file in list_file:
    if 'Cell9_' in file:
        df_cap = pd.read_csv('{folder}\{status}\{file}'.format(folder=path, status=status, file=file))
        soh = df_cap['capacity'].mean() / cap_n * 100
        flag = abs(soh_label - soh) < eps
        if not np.any(abs(soh_label - soh) < eps):
            continue
        label = soh_label[flag][0]
        # exclude duplicated data
        if label in list_legend_label:
            continue
        data = df_cap['deltCap'].fillna(0)
        ax.plot(np.arange(len(data)) + 1, data.values, label='Group_1_{0}%SOH'.format(label), color=colors[cnt],
                marker=markers[cnt])
        list_legend_label.append(label)
        list_legend_idx.append(cnt)
        cnt += 1
list_legend_idx.sort(key=lambda idx: list_legend_label[idx])

handles, label = plt.gca().get_legend_handles_labels()
plt.legend([handles[item] for item in list_legend_idx], [label[item] for item in list_legend_idx], fontsize=legend_size)
ax.set_ylim(0, 0.15)
ax.set_xlim(0, 90)
ax.set_ylabel('Delta Capacity / $A \cdot h$', fontsize=axis_font)
ax.set_xlabel('Voltage segment index', fontsize=axis_font)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
ax.grid()

# plot training result
# dchg partial sequence
path = r'./training_result_2023_07_19_14_14_38_56'
df_history = pd.read_csv('{folder}\{file}'.format(folder=path, file='history.csv'))
df_metric = pd.read_csv('{folder}\{file}'.format(folder=path, file='metric.csv'))
len_training = 160
len_validation = 424
epochs = 500

ax.clear()
cnt = 0
list_legend = []
ax.clear()
ax.plot(np.arange(len(df_history)) + 1, df_history['loss'].values, label='training loss', color=colors[cnt],
        marker=markers[cnt])
cnt += 1
ax.plot(np.arange(len(df_history)) + 1, df_history['val_loss'].values, label='validation loss', color=colors[cnt],
        marker=markers[cnt])
cnt += 1
ax.set_ylim(0, 1.5)
ax.set_xlim(0, 500)
ax.set_ylabel('Loss', fontsize=axis_font)
ax.set_xlabel('Epochs', fontsize=axis_font)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
ax.grid()
ax.legend(fontsize=legend_size)

cnt = 0
train_rmse = []
train_mae = []
val_rmse = []
val_mae = []
for epoch in range(epochs):
    df_temp = df_metric[df_metric['epochs'] == epoch + 1]
    # training part calculate
    train_mae.append(df_temp['dev_soh'].iloc[:len_training].values.mean())
    train_rmse.append(np.sqrt(np.mean(df_temp['dev_soh'].iloc[:len_training].values ** 2)))
    # validation part calculate
    val_mae.append(df_temp['dev_soh'].iloc[len_training:].values.mean())
    val_rmse.append(np.sqrt(np.mean(df_temp['dev_soh'].iloc[len_training:].values ** 2)))

ax.clear()
ax.plot(np.arange(len(train_mae)) + 1, np.array(train_mae), label='training MAE',
        color=colors[cnt], marker=markers[cnt])
cnt += 1
ax.plot(np.arange(len(val_mae)) + 1, np.array(val_mae), label='validation MAE',
        color=colors[cnt], marker=markers[cnt])
cnt += 1

ax.set_ylim(0, 7)
ax.set_xlim(0, 500)
ax.set_ylabel('Error / %', fontsize=axis_font)
ax.set_xlabel('Epochs', fontsize=axis_font)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
ax.grid()
ax.legend(fontsize=legend_size)

ax.clear()
cnt = 0
ax.plot(np.arange(len(train_rmse)) + 1, np.array(train_rmse), label='training RMSE',
        color=colors[cnt], marker=markers[cnt])
cnt += 1
ax.plot(np.arange(len(val_rmse)) + 1, np.array(val_rmse), label='validation RMSE',
        color=colors[cnt], marker=markers[cnt])
cnt += 1
ax.grid()
ax.set_ylim(0, 7)
ax.set_xlim(0, 500)
ax.set_ylabel('Error / %', fontsize=axis_font)
ax.set_xlabel('Epochs', fontsize=axis_font)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
ax.legend(fontsize=legend_size)

train_rmse = []
train_mae = []
ax.clear()
for idx_peak in range(4):
    val_rmse = []
    val_mae = []
    for epoch in range(epochs):
        df_temp = df_metric[df_metric['epochs'] == epoch + 1].iloc[len_training:].iloc[idx_peak::4]
        # validation part calculate
        val_mae.append(df_temp['dev_soh'].values.mean())
        val_rmse.append(np.sqrt(np.mean(df_temp['dev_soh'].values ** 2)))
    ax.plot(np.arange(len(val_mae)) + 1, np.array(val_mae), label='validation MAE for Peak{0}'.format(idx_peak + 1),
            color=colors[idx_peak], marker=markers[idx_peak])

# joint partial sequence
path = r'./training_result_2023_07_20_22_22_12_56'
df_history = pd.read_csv('{folder}\{file}'.format(folder=path, file='history.csv'))
df_metric = pd.read_csv('{folder}\{file}'.format(folder=path, file='metric.csv'))
len_training = 320
len_validation = 848
chrg_idx = [range(128, 256), range(348, 440), range(564, 688), range(768, 848)]
dchg_idx = [range(0, 128), range(256, 348), range(440, 564), range(688, 768)]
# analysis at specified epoch
epoch = 300
label_range = ['95% <=SOH< 100%', '90% <=SOH< 95%', '85% <=SOH< 90%', '80% <=SOH< 85%']
# errorbar
ax.clear()
for idx_peak in range(4):
    df_temp = [df_metric[df_metric['epochs'] == epoch].iloc[len_training:].iloc[idx_log].iloc[idx_peak::4]
               for idx_log in chrg_idx]
    cnt_input_label = [[], [], [], []]
    for item in df_temp:
        for idx in range(soh_label.shape[0] - 1):
            cnt_input_label[idx].append(item[(item['true_soh'] >= soh_label[idx + 1]) &
                                             (item['true_soh'] < soh_label[idx])])
    cnt_input_label = [pd.concat(item) for item in cnt_input_label]
    mean_dev = [np.mean(item['pred_soh'] - item['true_soh']) for item in cnt_input_label]
    std_dev = [np.sqrt(np.mean(item['dev_soh'] ** 2)) for item in cnt_input_label]
    ax.scatter(label_range, mean_dev, label='Peak{0}'.format(idx_peak + 1), marker=markers[idx_peak])
ax.grid()
ax.set_ylim(-3.0, 3.0)
ax.set_ylabel('Error / %', fontsize=axis_font)
ax.set_xlabel('Label SOH range', fontsize=axis_font)
ax.legend(fontsize=legend_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
plt.legend(loc='upper left')

plt.errorbar(label_range, mean_dev, yerr=std_dev, elinewidth=4, capsize=4, fmt='o',
             label='Peak{0}'.format(idx_peak + 1))
