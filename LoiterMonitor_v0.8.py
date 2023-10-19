#LoiterMonitor_v0.8
# rearrange from v0.7

import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib import patches
from PIL import Image
import pandas as pd
import math
from tqdm import tqdm
import glob
import MeasureIsPower as mip

########
# setting #
########
vgene_sw = 1
experiment = 'gh' #no: novel object test, epm: elevated plus maze, of: open field, tc: three chumber, gh: grouphousing, hdh: habt. dishabit.

setting_dict = {
    'group': 'GroupDeva',
    'subject': 'D1',
    'day': 'Day01',
    'parts': 'upperbody',
    'fix_num': '100',
    'model_group': 'GroupDeva',
    'model_subject': 'D1',
    'csvoffset': '1'
}


home = os.path.expanduser("~")
data_path = '/Dropbox/B03_PopEco/Rat/rattipot/2nd/'+setting_dict['group']+'/'+setting_dict['day']+'/'
video_path = glob.glob('/Volumes/Extreme SSD/rat_society/hakataya2023/'+setting_dict['group']+'_'+setting_dict['day']+'*.mp4')
#video_path = glob.glob('/mnt/data_complex/syncbox/rat_analysis/'+setting_dict['group']+'/'+setting_dict['day']+'/'+setting_dict['group']+'*.mp4')
video_output = 'video/'
#video_output = '/mnt/data_complex/syncbox/rat_analysis/'+setting_dict['group']+'/'+setting_dict['day']+'/Video/'

locomotion_param = {
    'loc_lim': 1000,    #locomotion (mm) per second
    'loc_thre': 5,      #mm per second
    'loc_interval': 1,
    'freeze_thre': 10,  #second
    'sleep_thre': 100   #second
}

coords_columns=['frame', 'nose_x', 'nose_y', 'nose_likeli', 'head_x', 'head_y', 'head_likeli',
                'earR_x', 'earR_y', 'earR_likeli', 'earL_x', 'earL_y', 'earL_likeli',
                'neck_x', 'neck_y', 'neck_likeli', 'upperbody_x', 'upperbody_y','upperbody_likeli',
                'lowerbody_x', 'lowerbody_y','lowerbody_likeli', 'tailbase_x', 'tailbase_y','tailbase_likeli']



############
# main process #
############

#change current directory
os.chdir(home+data_path)


#any profile loading
vp, ap, area_xy, area_path = mip.profile_load()
fps = vp['fps'][0]
frame_max = vp['frames'][0]
start_frame = vp['start of frame'][0]
end_frame = vp['end of frame'][0]
sc = vp['scale'][0]
area_col = ap.columns
area_ind = ap.index

#coordination & locomotion file setting
coords_data = mip.coords_file_load(coords_columns, **setting_dict)
coords_x = coords_data[setting_dict['parts']+'_x']
coords_y = coords_data[setting_dict['parts']+'_y']

#locomotion data calcuration
df_locomotion, dict_iteration = mip.locomotion_calc(coords_x, coords_y, frame_max, fps, sc, locomotion_param)

#entry&staying area calcuration
entry, staying_time, itr_entry, itr_staying = mip.entry_calc(coords_x, coords_y, ap, area_path, frame_max, fps)

#
for i, col in enumerate(ap.columns):
    df_locomotion['entry to '+col] = entry[i]
    dict_iteration['itr_entry_'+col] = itr_entry[i]
for i, col in enumerate(ap.columns):
    staying_time[i] /= fps #convert frames -> seconds
    df_locomotion['time in '+col+'(sec.)'] = staying_time[i]
    dict_iteration['itr_staying_'+col] = itr_staying[i]

print(df_locomotion)
df_locomotion.to_csv(setting_dict['subject']+'_'+setting_dict['day']+'_loiter_'+setting_dict['parts']+'.csv')

if vgene_sw:
    mip.plot_check(coords_x, coords_y, ap, vp, video_path[0], area_xy, setting_dict['subject'], locomotion_param['loc_interval'], dict_iteration, video_output)




#以下、そのうちGUIにする用
'''
if __name__ == "__main__":
    #Main window
    main = tk.Tk()
    main.title("LoiterMonitor v0.8")
    main.geometory("1280x800")

    #Tab
    nb = ttk.Notebook(main) #tab instance
    tab1 = tk.Frame(nb)
    tab2 = tk.Frame(nb)
    nb.add(tab1, text='1st Tab', padding=4)
    nb.add(tab2, text='2nd Tab', padding=4)
    lbl1 = tk.Label(master=tab1, text='here is tab1')
    lbl2 = tk.Label(master=tab2, text='here is tab2')

    nb.pack(expand=1, fill='both')
    lbl1.pack()
    lbl2.pack()

    btn = tk.Button()

    main.mainloop()
'''
