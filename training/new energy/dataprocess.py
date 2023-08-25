import pandas as pd
import numpy as np
import scipy.interpolate as interplt
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.interpolate as interp

volt_seg = np.arange(2.5, 4.2, 0.02)
TIME_INTERVAL = 60  # 1 min

def wrapper_interp(x, y, xin, kind='linear'):
    try:
        ans = interp.interp1d(x, y, bounds_error=False, fill_value=np.nan, kind=kind)(xin)
    except Exception as e:
        print(e.message)
    return ans

def readcsvdata(file_name, capacity, R_in):
    df_data = pd.read_csv(file_name)
    df_data['ocv'] = df_data['voltage'] - df_data['current'] * R_in  # Rint model
    df_data['capacity'] = capacity
    df_data['index'] = np.arange(len(df_data)) + 1
    df_chrg = df_data[(df_data['capacity_chrg'].diff() > 0)]
    df_dchg = df_data[(df_data['capacity_dchg'].diff() > 0)]
    interp_cap_chrg = pd.Series(wrapper_interp(df_chrg['index'], df_chrg['capacity_chrg'], df_data['index']))
    interp_cap_chrg = interp_cap_chrg.fillna(method='bfill')
    interp_cap_chrg = interp_cap_chrg.fillna(method='ffill')
    interp_cap_dchg = pd.Series(wrapper_interp(df_dchg['index'], df_dchg['capacity_dchg'], df_data['index']))
    interp_cap_dchg = interp_cap_dchg.fillna(method='bfill')
    interp_cap_dchg = interp_cap_dchg.fillna(method='ffill')
    ans_chrg = pd.DataFrame()
    ans_dchg = pd.DataFrame()

    df_dchg = df_data[df_data['current'] < 0]
    interp_cap_dchg = interp_cap_dchg[df_data['current'] < 0]
    interp_capacity = pd.Series(wrapper_interp(df_dchg['ocv'].values, capacity - interp_cap_dchg.values, volt_seg))
    deltCap = interp_capacity.diff()
    ans_dchg['deltCap'] = deltCap
    ans_dchg['capacity'] = capacity

    return ans_chrg, ans_dchg


cap_n = 1.1
soc_label = np.linspace(1, 100, 100)
data_ocv = pd.read_csv('OCV_SoC.csv')
folder_list = [r'D:\BMS\data\new energy\2018-04-12_batchdata_updated_struct_errorcorrect']
for path in folder_list:
    list_summary = os.listdir(r'{0}/summary'.format(path))
    list_file = os.listdir(r'{0}/data'.format(path))
    for item in enumerate(list_summary):
        file_summary = item[1]
        barcode = file_summary.split('_')[0]
        df_summary = pd.read_csv(r'{0}/summary/{1}'.format(path, file_summary))
        for cycle in df_summary['cycle'].values:
            file_name = '{0}_cycle_{1}.csv'.format(barcode, cycle)
            if not file_name in list_file:
                print('no file: {0}'.format(file_name))
                continue
            capacity = df_summary[df_summary['cycle'] == cycle]['QDischarge'].values[0]
            R_in = df_summary[df_summary['cycle'] == cycle]['IR'].values[0]
            with open(r'{0}/data/{1}'.format(path, file_name)) as file:
                df_chrgdata, df_dchgdata = readcsvdata(file, capacity, R_in)
            if not os.path.exists(r'{0}/capInloop/chrg'.format(path)):
                os.makedirs(r'{0}/capInloop/chrg'.format(path))
            df_chrgdata.to_csv(r'{0}/capInloop/chrg/{1}'.format(path, file_name))
            if not os.path.exists(r'{0}/capInloop/dchg'.format(path)):
                os.makedirs(r'{0}/capInloop/dchg'.format(path))
            df_dchgdata.to_csv(r'{0}/capInloop/dchg/{1}'.format(path, file_name))
        print('progress: {0}/{1}'.format(item[0] + 1, len(list_summary)))
print('finished')
