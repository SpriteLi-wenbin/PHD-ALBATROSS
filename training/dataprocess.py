import os
import re
import pandas as pd
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def wrap_interplt(x, y, xin):
    ans = interp.interp1d(x, y, bounds_error=False, fill_value=np.nan)(xin)
    if np.all(np.isnan(ans)):
        print('nan')

    return ans


# group into continuous monotonous series, decreasing or increasiong
def group_to_cont(data, mode='dec'):
    diff = [-1, 0.001]
    if mode == 'inc':
        diff = [-0.001, 1]

    temp = np.diff(data)
    ans = np.cumsum(~((temp < diff[1]) & (temp > diff[0])))

    return ans


def extract_ocv_soc(voltage, soc):
    label_soc = np.linspace(1, 100, 100)
    intplt_ocv = wrap_interplt(soc, voltage, label_soc)
    ans = pd.DataFrame({'ocv': np.array(intplt_ocv), 'soc': label_soc})
    return ans


def extract_cap_seq(voltage, cap, volt_seg):
    intplt_cap = wrap_interplt(voltage, cap, volt_seg)
    intplt_cap = np.diff(intplt_cap)
    intplt_cap = np.nan_to_num(intplt_cap)

    return np.array(intplt_cap)


def sort_key(name):
    name = name.split('.')[0]
    if len(name.split('_')) > 1:
        num = name.split('_')[1]
    else:
        num = 'error'
    if num.isdigit():
        num = int(num)
    else:
        num = -1
    return name.split('_')[0], num


path = r'D:\BMS\data\oxford\Path dependent battery degradation dataset'
list_folder = ['Group_1_csv', 'Group_2_csv', 'Group_3_csv', 'Group_4_csv']
list_folder = ['Group_1_csv', 'Group_2_csv', 'Group_3_csv']
volt_seg = np.arange(2.5, 4.2, 0.02)
pattern = re.compile('.*.csv')

'''
# rename file to Cellxx_yy.csv,  x stands for cell Nr., y stands for file Nr.
for folder in list_folder:
    for file in os.listdir(r'{0}\{1}'.format(path, folder)):
        file_path = r'{0}\{1}\{2}'.format(path, folder, file)
        if not os.path.isfile(file_path):
            continue
        temp = file.split('-')
        cell_name = temp[-1].split('.')[0]
        cell_name = ''.join(cell_name.split(' '))
        if len(temp[0].split('.')) < 2:
            nr_file = '1'
        else:
            nr_file = temp[0].split('.')[1:]
            nr_file = ''.join(nr_file[-1].split(' '))
        ans = '{0}_{1}.csv'.format(cell_name, nr_file)
        os.rename(file_path, r'{0}\{1}\{2}'.format(path, folder, ans))
'''
'''
# change capacity in charge rpt cycle to the one in discharge rpt cycle
for folder in list_folder:
    path_chrg = r'{0}\{1}\{2}\chrg'.format(path, folder, r'rpt')
    path_dchg = r'{0}\{1}\{2}\dchg'.format(path, folder, r'rpt')
    list_file_chrg = os.listdir(path_chrg)
    list_file_dchg = os.listdir(path_dchg)
    for file in list_file_chrg:
        if not re.match(pattern, file):
            print(r'no match: {0}\{1}'.format(path_chrg, file))
            continue
        if file not in list_file_dchg:
            print(r'no discharge rpt: {0}\{1}'.format(path_chrg, file))
            continue
        if os.path.isfile(r'{0}\{1}'.format(path_chrg, file)):
            df_data_chrg = pd.read_csv(r'{0}\{1}'.format(path_chrg, file))
            df_data_dchg = pd.read_csv(r'{0}\{1}'.format(path_dchg, file))
        else:
            continue
        df_data_chrg['capacity_chrg'] = df_data_chrg['capacity']
        df_data_chrg['capacity'] = df_data_dchg['capacity']
        df_data_chrg.to_csv(r'{0}\{1}'.format(path_chrg, file))
    print('{0} finish'.format(folder))

'''
for folder in list_folder:
    list_file = os.listdir(r'{0}\{1}'.format(path, folder))
    list_file.sort(key=sort_key)
    path_save = r'{0}\{1}\{2}\chrg'.format(path, folder, r'soc_ocv')
    #path_save = r'{0}\{1}\{2}\chrg'.format(path, folder, r'rpt')
    num_file = len(list_file)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    print('begin folder: {0}'.format(folder))
    for file in list_file:
        if os.path.isfile(r'{0}\{1}\{2}'.format(path, folder, file)):
            df_data = pd.read_csv(r'{0}\{1}\{2}'.format(path, folder, file))
        else:
            continue
        df_data_dchg = df_data[df_data['Step'] == 8]
        df_data_chrg = df_data[df_data['Step'] == 10]
        df_data_rpt = df_data[df_data['Step'] == 6]  # CC-CV charge at C/2 to 2.5V (step 6)
        #df_data_rpt = df_data[df_data['Step'] == 4]  # CC-CV discharge at C/2 to 2.5V (step 4)
        cnt_dchg = 0
        cnt_chrg = 0
        cnt_rpt = 0
        print('begin file: {0}'.format(file))

        # chrg ocv_soc
        df_data = df_data_chrg[['TestTime', 'Amphr', 'Volts', 'Amps']]
        groups = group_to_cont(df_data['TestTime'], 'inc')
        set_groups = list(set(groups))
        set_groups.sort()
        for item in enumerate(set_groups):
            group = item[1]
            df_temp = df_data.iloc[1:].iloc[groups == group]
            if (df_temp['Volts'].max() < 4.19) or (abs(df_temp['Amps'].mean()) < 0.1):
                continue
            cap_n = df_temp['Amphr'].max()
            df_temp['soc'] = df_temp['Amphr'] / cap_n * 100
            soc_ocv = extract_ocv_soc(df_temp['Volts'].values, df_temp['soc'].values)
            # intplt_ocv = soc_ocv['ocv']
            # intplt_cap = extract_cap_seq(soc_ocv['ocv'], soc_ocv['soc'] / 100 * cap_n, volt_seg)
            soc_ocv = soc_ocv.fillna(method='bfill')
            soc_ocv = soc_ocv.fillna(method='ffill')
            soc_ocv['capacity'] = cap_n
            soc_ocv.to_csv(r'{0}\{1}_{2}.csv'.format(path_save, file.split('.')[0], str(cnt_chrg)))
            cnt_chrg += 1
'''
        #  rpt data
        df_data = df_data_rpt[['TestTime', 'StepTime', 'Amphr', 'Volts', 'Amps']]
        groups = group_to_cont(df_data['TestTime'], 'inc')
        set_groups = list(set(groups))
        set_groups.sort()
        for item in enumerate(set_groups):
            group = item[1]
            df_temp = df_data.iloc[1:].iloc[groups == group]
            #if (df_temp['Volts'].min() > 2.51) or (abs(df_temp['Amps'].mean()) < 0.1):
            if (df_temp['Volts'].max() < 4.19) or (abs(df_temp['Amps'].mean()) < 0.1):
                continue
            cap_n = df_temp['Amphr'].max()
            df_temp['capacity'] = cap_n
            df_temp.to_csv(r'{0}\{1}_{2}.csv'.format(path_save, file.split('.')[0], str(cnt_rpt)))
            cnt_rpt += 1


        # dchg ocv_soc
        df_data = df_data_dchg[['TestTime', 'StepTime', 'Amphr', 'Volts', 'Amps']]
        groups = group_to_cont(df_data['TestTime'], 'inc')
        set_groups = list(set(groups))
        set_groups.sort()
        for item in enumerate(set_groups):
            group = item[1]
            df_temp = df_data.iloc[1:].iloc[groups == group]
            if (df_temp['Volts'].min() > 2.51) or (abs(df_temp['Amps'].mean()) < 0.1):
                continue
            cap_n = df_temp['Amphr'].max()
            df_temp['soc'] = 100 - df_temp['Amphr'] / cap_n * 100
            soc_ocv = extract_ocv_soc(df_temp['Volts'].values, df_temp['soc'].values)
            #intplt_ocv = soc_ocv['ocv']
            #intplt_cap = extract_cap_seq(soc_ocv['ocv'], soc_ocv['soc'] / 100 * cap_n, volt_seg)
            soc_ocv = soc_ocv.fillna(method='bfill')
            soc_ocv = soc_ocv.fillna(method='ffill')
            soc_ocv['capacity'] = cap_n
            soc_ocv.to_csv(r'{0}\{1}_{2}.csv'.format(path_save, file.split('.')[0], str(cnt_dchg)))
            cnt_dchg += 1
'''


