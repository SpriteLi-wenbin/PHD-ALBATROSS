import os
import keras
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import keras.layers as layers
from keras import Model


def wrap_loss_func(y_true, y_pred):
    loss_func = keras.losses.kld
    loss_val = loss_func(y_true, y_pred)

    return loss_val

def new_soh_model_single_peak(num_seq, time_step=1, DROPOUT=0, DROPOUT_RECUR=0, lstm_layer_units=128,
                              model_name='soh_algorithm'):
    input_cap_seq = layers.Input(shape=(time_step, num_seq,), name='capacity sequence')
    input_work_mode = layers.Input(shape=(time_step, 2,), name='charging/discharging mode')
    input_peak_mode = layers.Input(shape=(time_step, 4,), name='peak vector')
    # masking layer for zero input
    # masking = layers.Masking()
    input_merge_peak_vector = layers.concatenate([input_cap_seq, input_peak_mode])
    hid_layer_cap = layers.Dense(units=32, activation=tf.nn.tanh)(input_merge_peak_vector)

    input_merge_work_mode = layers.concatenate([hid_layer_cap, input_work_mode])
    hid_layer_cap_halfpeak = layers.Dense(units=64, activation=tf.nn.tanh)(input_merge_work_mode)

    lstm_layer = layers.LSTM(units=lstm_layer_units, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid,
                             dropout=DROPOUT, recurrent_dropout=DROPOUT_RECUR)(hid_layer_cap_halfpeak)
    # softmax for multi-class
    output_layer = layers.Dense(units=5, activation=tf.nn.softmax, name='category_result')(lstm_layer)
    # sigmoid for multi-label
    # output_layer = layers.Dense(units=5, activation=tf.nn.sigmoid, name='category_result')(lstm_layer)
    soh_model = Model(inputs=[input_cap_seq, input_peak_mode, input_work_mode], outputs=[output_layer], name=model_name)
    return soh_model


def metric_soh(y_true, y_pred):
    pred = y_pred.numpy()
    truth = y_true.numpy()
    # print(pred)
    # print(truth)
    base = [1.0, 0.95, 0.9, 0.85, 0.8]
    for idx in range(len(y_pred)):
        for j in range(len(y_pred[idx])):
            log_data['pred_cat{0}'.format(j)].append(pred[idx][j])
            log_data['truth_cat{0}'.format(j)].append(truth[idx][j])
    soh_true = (truth * base).sum()
    soh_pred = (pred * base).sum()
    return (soh_pred - soh_true) * 100


def cal_soh_label(soh):
    base = [1.0, 0.95, 0.9, 0.85, 0.8]
    ans = [0, 0, 0, 0, 0]

    if soh > base[0]:
        return [1, 0, 0, 0, 0]
    if soh < base[-1]:
        return [0, 0, 0, 0, 1]

    for idx in range(1, len(base)):
        if soh < base[idx]:
            continue
        ans[idx - 1] = (soh - base[idx]) / (base[idx - 1] - base[idx])
        ans[idx] = 1 - ans[idx - 1]
        break
    return ans


def extract_training_data(file, cap_seg, status):
    df_data = pd.read_csv(file)
    if 'capacity' in df_data.columns:
        capacity = df_data['capacity'].mean()
    else:
        capacity = np.nan
    peak_indicator = [np.zeros(len(cap_seg)) for item in cap_seg]
    mode = {'chrg': np.array([0, 1]), 'dchg': np.array([1, 0])}
    cap_seq = []
    work_mode = []
    for item in enumerate(cap_seg):
        peak_indicator[item[0]][item[0]] = 1
        temp = df_data['deltCap'].values[item[1]]
        temp = np.nan_to_num(temp)
        cap_seq.append(np.clip(temp / cap_baseline[status][item[0]], 0.0, 1.0))
        work_mode.append(mode[status])

    return cap_seq, peak_indicator, work_mode, capacity


def find_closest(src, dst):
    diff = np.absolute(np.array(src) - np.array(dst))
    idx = np.argmin(diff)
    return src[idx]


num_seq = 10
volt_seg = np.arange(2.5, 4.2, 0.02)
soh_label = np.array([100, 95, 90, 85, 80])
cap_n = 3.0
list_folder = ['Group_1_csv', 'Group_2_csv', 'Group_3_csv', 'Group_4_csv']
path = r'D:\BMS\data\oxford\Path dependent battery degradation dataset'
cap_seg = {'chrg': [range(40, 50), range(48, 58), range(52, 62), range(65, 75)],
           'dchg': [range(35, 45), range(45, 55), range(52, 62), range(59, 69)]}
cap_baseline = {'chrg': [0.1227297099238634, 0.110733574218129, 0.1110255611317823, 0.0735948090152245],
                'dchg': [0.0542762291037793, 0.1100274017008113, 0.1100274017008113, 0.074525662291244]}

log_data = {}  # initialize log
for idx in range(5):
    log_data['pred_cat{0}'.format(idx)] = []
    log_data['truth_cat{0}'.format(idx)] = []
soh_model = new_soh_model_single_peak(num_seq=num_seq)  # link layers
metrics = [keras.metrics.categorical_accuracy, metric_soh]
soh_model.compile(loss=keras.losses.kl_divergence, optimizer='adam', metrics=metrics, run_eagerly=True)

shape_input = soh_model.input_shape
shape_output = soh_model.output_shape

test_input = []
test_peak = []
test_label = []
test_workmode = []
validation_input = []
validation_peak = []
validation_label = []
validation_workmode = []
log_src_training = []
log_src_validation = []

for folder in list_folder:
    # dchg
    path_dataset = r'{0}\{1}\capInLoop\dchg'.format(path, folder)
    list_trainfile = os.listdir(path_dataset)
    for file in list_trainfile:
        path_file = r'{0}\{1}'.format(path_dataset, file)
        with open(path_file) as trainfile:
            temp_cap_seq, temp_peak, temp_workmode, temp_capacity = extract_training_data(trainfile, cap_seg['dchg'],
                                                                                          'dchg')
        soh = temp_capacity / cap_n
        validation_output_label = cal_soh_label(soh)
        train_output_label = find_closest(soh_label, soh * 100)
        train_output_label = cal_soh_label(train_output_label / 100)

        train_output_label = [np.array(train_output_label).reshape(shape_output[1:]) for item in temp_cap_seq]
        validation_output_label = [np.array(validation_output_label).reshape(shape_output[1:]) for item in temp_cap_seq]
        temp_cap_seq = [item.reshape(shape_input[0][1:]) for item in temp_cap_seq]
        temp_peak = [item.reshape(shape_input[1][1:]) for item in temp_peak]
        temp_workmode = [item.reshape(shape_input[2][1:]) for item in temp_workmode]

        for item in temp_cap_seq:
            if np.isnan(item.min()) or np.isnan(item.max()):
                print(path_file)

        if np.any(abs(soh_label - soh * 100) < 0.2):
            test_input.extend(temp_cap_seq)
            test_peak.extend(temp_peak)
            test_label.extend(train_output_label)
            test_workmode.extend(temp_workmode)
            log_src_training.append('{0}\{1}'.format(folder, file))
        elif np.any(abs(soh_label - soh * 100) < 0.8):
            test_input.extend(temp_cap_seq)
            test_peak.extend(temp_peak)
            test_label.extend(validation_output_label)
            test_workmode.extend(temp_workmode)
            log_src_training.append('{0}\{1}'.format(folder, file))
        else:
            validation_input.extend(temp_cap_seq)
            validation_peak.extend(temp_peak)
            validation_label.extend(validation_output_label)
            validation_workmode.extend(temp_workmode)
            log_src_validation.append('{0}\{1}'.format(folder, file))

    # chrg
    path_dataset = r'{0}\{1}\capInLoop\chrg'.format(path, folder)
    list_trainfile = os.listdir(path_dataset)
    for file in list_trainfile:
        path_file = r'{0}\{1}'.format(path_dataset, file)
        with open(path_file) as trainfile:
            temp_cap_seq, temp_peak, temp_workmode, temp_capacity = extract_training_data(trainfile, cap_seg['chrg'],
                                                                                          'chrg')
        soh = temp_capacity / cap_n
        validation_output_label = cal_soh_label(soh)
        train_output_label = find_closest(soh_label, soh * 100)
        train_output_label = cal_soh_label(train_output_label / 100)

        train_output_label = [np.array(train_output_label).reshape(shape_output[1:]) for item in temp_cap_seq]
        validation_output_label = [np.array(validation_output_label).reshape(shape_output[1:]) for item in temp_cap_seq]
        temp_cap_seq = [item.reshape(shape_input[0][1:]) for item in temp_cap_seq]
        temp_peak = [item.reshape(shape_input[1][1:]) for item in temp_peak]
        temp_workmode = [item.reshape(shape_input[2][1:]) for item in temp_workmode]

        for item in temp_cap_seq:
            if np.isnan(item.min()) or np.isnan(item.max()):
                print(path_file)

        if np.any(abs(soh_label - soh * 100) < 0.2):
            test_input.extend(temp_cap_seq)
            test_peak.extend(temp_peak)
            test_label.extend(train_output_label)
            test_workmode.extend(temp_workmode)
            log_src_training.append('{0}\{1}'.format(folder, file))
        elif np.any(abs(soh_label - soh * 100) < 0.8):
            test_input.extend(temp_cap_seq)
            test_peak.extend(temp_peak)
            test_label.extend(validation_output_label)
            test_workmode.extend(temp_workmode)
            log_src_training.append('{0}\{1}'.format(folder, file))
        else:
            validation_input.extend(temp_cap_seq)
            validation_peak.extend(temp_peak)
            validation_label.extend(validation_output_label)
            validation_workmode.extend(temp_workmode)
            log_src_validation.append('{0}\{1}'.format(folder, file))



test_label = np.array(test_label)
test_input = np.array(test_input)
test_peak = np.array(test_peak)
test_workmode = np.array(test_workmode)
validation_input = np.array(validation_input)
validation_peak = np.array(validation_peak)
validation_label = np.array(validation_label)
validation_workmode = np.array(validation_workmode)

validation_data = ([validation_input, validation_peak, validation_workmode], validation_label)

training_history = soh_model.fit(x=[test_input, test_peak, test_workmode], y=test_label,
                                 epochs=500, shuffle=False, verbose=1, validation_data=validation_data, callbacks=None)

save_path = r'.\training_result_{0}'.format(time.strftime('%Y_%m_%d_%H_%H_%M_%S'))
if not os.path.exists(save_path):
    os.mkdir(save_path)
df_history = pd.DataFrame(training_history.history)
df_log = pd.DataFrame(log_data)
df_soh = df_log[['pred_cat0', 'pred_cat1', 'pred_cat2', 'pred_cat3', 'pred_cat4']] * soh_label
df_log['pred_soh'] = np.sum(df_soh.values, axis=1)
df_soh = df_log[['truth_cat0', 'truth_cat1', 'truth_cat2', 'truth_cat3', 'truth_cat4']] * soh_label
df_log['true_soh'] = np.sum(df_soh.values, axis=1)
df_log['dev_soh'] = np.abs(df_log['true_soh'] - df_log['pred_soh'])
df_log['epochs'] = np.arange(len(df_log)) // (len(test_input) + len(validation_input)) + 1

df_log.to_csv(r'{0}\{1}'.format(save_path, 'metric.csv'))
df_history.to_csv(r'{0}\{1}'.format(save_path, 'history.csv'))
with open(r'{0}\{1}'.format(save_path, 'readme.txt'), 'w') as file:
    content = 'epochs:{0}\ntest size: {1}\nvalidation size: {2}\nlog train: {3}\nlog validation: {4}'
    file.write(content.format(500, len(test_input), len(validation_input), log_src_training, log_src_validation))
