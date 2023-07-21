import os
import pandas as pd
import numpy as np
import re
import scipy.interpolate as interp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wrap_interplt(x, y, xin):
    ans = interp.interp1d(x, y, bounds_error=False, fill_value=np.nan)(xin)
    if np.all(np.isnan(ans)):
        print('nan')

    return ans


volt_seg = np.arange(2.5, 4.2, 0.02)

list_folder = ['Group_1_csv', 'Group_2_csv', 'Group_3_csv', 'Group_4_csv']
path = r'D:\BMS\data\oxford\Path dependent battery degradation dataset'

'''
# discharge
for folder in list_folder:
    path_rptfile = r'{0}\{1}\rpt\dchg'.format(path, folder)
    path_sococv = r'{0}\{1}\soc_ocv\dchg'.format(path, folder)
    path_save = r'{0}\{1}\capInLoop\dchg'.format(path, folder)
    if not os.path.exists(path_save):
        os.makedirs(path_save, exist_ok=True)
    list_rptfile = os.listdir(path_rptfile)
    list_ocvfile = os.listdir(path_sococv)
    for rptfile in list_rptfile:
        if not os.path.isfile(r'{0}\{1}'.format(path_rptfile, rptfile)):
            continue
        pattern = re.compile('.*.csv')
        if not re.match(pattern, rptfile):
            continue
        sample_name = rptfile[:-6]
        df_rptdata = pd.read_csv(r'{0}\{1}'.format(path_rptfile, rptfile))  # capacity test with nominal condition
        cap_n = df_rptdata['capacity'].mean()
        df_rptdata['soc'] = np.clip(100 - abs(df_rptdata['Amphr'] / cap_n) * 100, 0, 100)
        # get corresponding ocv data
        if rptfile in list_ocvfile:
            df_ocvdata = pd.read_csv(r'{0}\{1}'.format(path_sococv, rptfile))
        else:
            continue
        # get ocv
        interp_ocv = wrap_interplt(df_ocvdata['soc'], df_ocvdata['ocv'], df_rptdata['soc'])
        df_rptdata['ocv'] = interp_ocv
        # get capacity sequence
        deltCap = wrap_interplt(df_rptdata['ocv'], abs(df_rptdata['soc'] * df_rptdata['capacity'] / 100), volt_seg)
        deltCap = np.diff(deltCap)
        ans = pd.DataFrame({'deltCap': deltCap})
        ans['capacity'] = cap_n
        ans.to_csv(r'{0}\{1}'.format(path_save, rptfile))  # save file

'''
#charge
for folder in list_folder:
    path_rptfile = r'{0}\{1}\rpt\chrg'.format(path, folder)
    path_sococv = r'{0}\{1}\soc_ocv\chrg'.format(path, folder)
    path_save = r'{0}\{1}\capInLoop\chrg'.format(path, folder)
    if not os.path.exists(path_save):
        os.makedirs(path_save, exist_ok=True)
    list_rptfile = os.listdir(path_rptfile)
    list_ocvfile = os.listdir(path_sococv)
    print('begin folder:{0}'.format(folder))
    for rptfile in list_rptfile:
        if not os.path.isfile(r'{0}\{1}'.format(path_rptfile, rptfile)):
            continue
        pattern = re.compile('.*.csv')
        if not re.match(pattern, rptfile):
            continue
        print('begin file:{0}'.format(rptfile))
        sample_name = rptfile[:-6]
        df_rptdata = pd.read_csv(r'{0}\{1}'.format(path_rptfile, rptfile))  # capacity test with nominal condition
        cap_n = df_rptdata['capacity'].mean()
        df_rptdata['soc'] = np.clip(abs(df_rptdata['Amphr'] / cap_n) * 100, 0, 100)
        # get corresponding ocv data
        if rptfile in list_ocvfile:
            df_ocvdata = pd.read_csv(r'{0}\{1}'.format(path_sococv, rptfile))
        else:
            print('no matching ocv file: {0}\{1}'.format(path_rptfile, rptfile))
            continue
        # get ocv
        interp_ocv = wrap_interplt(df_ocvdata['soc'], df_ocvdata['ocv'], df_rptdata['soc'])
        df_rptdata['ocv'] = interp_ocv
        # get capacity sequence
        deltCap = wrap_interplt(df_rptdata['ocv'], abs(df_rptdata['soc'] * df_rptdata['capacity'] / 100), volt_seg)
        deltCap = np.diff(deltCap)
        ans = pd.DataFrame({'deltCap': deltCap})
        ans['capacity'] = cap_n
        ans.to_csv(r'{0}\{1}'.format(path_save, rptfile))  # save file

'''
# plot
list_folder = ['Group_1_csv', 'Group_2_csv', 'Group_3_csv', 'Group_4_csv']
cap_seg = {'chrg': [range(40, 50), range(48, 58), range(52, 62), range(65, 75)],
           'dchg': [range(35, 45), range(45, 55), range(52, 62), range(59, 69)]}
cap_n = 3.0
fig = plt.figure()
ax = plt.axes()
ans = []

for folder in list_folder:
    path_data = r'{0}\{1}\capInLoop\dchg'.format(path, folder)
    list_file = os.listdir(path_data)
    for file in list_file:
        if not os.path.isfile(r'{0}\{1}'.format(path_data, file)):
            continue
        pattern = re.compile('.*.csv')
        if not re.match(pattern, file):
            continue
        sample_name = file.split('_')[0]
        df_data = pd.read_csv(r'{0}\{1}'.format(path_data, file))
        soh = df_data['capacity'].mean() / cap_n * 100
        #ax.plot(np.nan_to_num(df_data['deltCap'].values), label='{0}:{1}%'.format(sample_name, soh))
        if abs(soh - 100) < 0.7:
            print(np.nansum(df_data['deltCap'].values))
            ans.append([np.max(df_data['deltCap'].values[item]) for item in cap_seg['dchg']])
'''