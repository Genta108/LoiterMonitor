#########################
### MeasureIsPower.Py ###
#########################
import os
import cv2
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib import patches
from tqdm import tqdm



def profile_load():
    #video profile setting
    vp = pd.read_csv('videoprofile.csv')

    #area profile setting
    ap = pd.read_csv('areaprofile.csv', index_col=0)

    area_xy = []
    area_path = []
    for i, col in enumerate(ap.columns): #columns loop
        area_xy.append([])
        for j in range(0, len(ap.index), 2):
            area_xy[i].append([ap[col][j],ap[col][j+1]])
        area_path.append(path.Path(area_xy[i]))


    return vp, ap, area_xy, area_path


def coords_file_load(coords_columns, **setting_dict):
    coords_file_name = glob.glob('*DLC_resnet50_'+setting_dict['model_group']+'_'+setting_dict['model_subject']+'*fixed'+setting_dict['fix_num']+'.csv')
    coords_data = pd.read_csv(coords_file_name[0], header=None, skiprows=int(setting_dict['csvoffset']))
    coords_data.columns = coords_columns

    return coords_data



def locomotion_calc(coords_x, coords_y, frame_max, fps, sc, locomotion_param):
    loc_lim = locomotion_param['loc_lim']
    loc_thre = locomotion_param['loc_thre']
    loc_interval = locomotion_param['loc_interval']
    freeze_thre = locomotion_param['freeze_thre']
    sleep_thre = locomotion_param['sleep_thre']

    locomotion = []
    velocity = []
    agility = []
    active = []

    freeze_count = freeze_thre * fps
    sleep_count = sleep_thre * fps

    sum_locomotion = 0
    sum_freezing = 0
    sum_sleeping = 0
    ave_velocity = 0
    std_velocity = 0
    ave_agility = 0
    std_agility = 0

    itr_locomotion = []
    itr_freezing = []
    itr_sleeping = []

    for f in tqdm(range(frame_max), desc='progress',leave=True):
        if (f%loc_interval)==0:
            loc_dx = 0
            loc_dy = 0
            if f == 0:
                loc_dx = coords_x[f+loc_interval]-coords_x[f]
                loc_dy = coords_y[f+loc_interval]-coords_y[f]
            else:
                loc_dx = coords_x[f]-coords_x[f-loc_interval]
                loc_dy = coords_y[f]-coords_y[f-loc_interval]

            loc_d = math.sqrt(loc_dx**2+loc_dy**2)*sc
            locomotion.append(loc_d)

            if loc_lim >= loc_d >= loc_thre:
                sum_locomotion += loc_d
                velocity.append(loc_d)
                agility.append(loc_d)
                freeze_count = freeze_thre * fps
                sleep_count = sleep_thre *fps
                active.append(1)
                itr_locomotion.append(sum_locomotion)
                itr_freezing.append(sum_freezing)
                itr_sleeping.append(sum_sleeping)
            else:
                velocity.append(0)
                active.append(0)
                freeze_count -= 1
                sleep_count -= 1
                if (freeze_count < 0) and (sleep_count >= 0):
                    sum_freezing += 1
                elif freeze_count == 0:
                    sum_freezing += freeze_thre * fps+1
                elif sleep_count == 0:
                    sum_fleezing -= sleep_thre * fps
                    sum_sleeping += sleep_thre * fps
                elif sleep_count < 0:
                    sum_sleeping += 1
                    freeze_count = freeze_thre * fps
                itr_locomotion.append(sum_locomotion)
                itr_freezing.append(sum_freezing)
                itr_sleeping.append(sum_sleeping)

    #velocity ave & std
    fmax = int(frame_max/loc_interval)
    ave_velocity = sum(velocity)/fmax
    for f in range(fmax):
        std_velocity += (velocity[f]-ave_velocity)**2
    std_velocity = np.sqrt(std_velocity/fps)

    #agility ave & std
    act = sum(active)
    ave_agility = sum(agility)/act
    for a in range(act):
        std_agility += (agility[a]-ave_agility)**2
    std_agility = np.sqrt(std_agility/act)

    sum_locomotion /= 1000 #mm -> m
    sum_freezing /= fps/loc_interval
    sum_sleeping /= fps/loc_interval

    df_locomotion = pd.DataFrame({'velocity ave. (mm/s)': [ave_velocity],
                       'velocity std. (mm/s)': [std_velocity],
                       'agility ave. (mm/s)': [ave_agility],
                       'agility std. (mm/s)': [std_agility],
                       'freezing (sec.)': [sum_freezing],
                       'sleeping (sec.)': [sum_sleeping],
                       'total locomation (m)': [sum_locomotion]
                      })

    dict_iteration = {
        'active': active,
        'itr_locomotion': itr_locomotion,
        'itr_freezing': itr_freezing,
        'itr_sleeping':itr_sleeping
    }

    return df_locomotion, dict_iteration



def entry_calc(coords_x, coords_y, ap, area_path, frame_max, fps):
    entry = np.zeros(len(ap.columns))
    inflg = np.zeros(len(ap.columns))
    staying_time = np.zeros(len(ap.columns))
    itr_entry = []
    itr_staying = []
    for a in ap.columns: #columns loop
        itr_entry.append([])
        itr_staying.append([])

    #entry&staying
    for f in tqdm(range(frame_max), desc='progress',leave=True):
        ex_area = 1
        parts_xy = [coords_x[f], coords_y[f]]

        for a in range(1, len(ap.columns)):
            if area_path[a].contains_point(parts_xy):
                ex_area = 0
                staying_time[a] += 1
                if inflg[a] == 0:
                    entry[a] += 1
                    inflg = np.zeros(len(ap.columns))
                    inflg[a] = 1
                break
        if ex_area:
            inflg = np.zeros(len(ap.columns))

        for a in range(1, len(ap.columns)):
            itr_staying[a].append(staying_time[a])
            itr_entry[a].append(entry[a])

    #exception for allrange
    for f in tqdm(range(frame_max), desc='progress',leave=True):
        parts_xy = [coords_x[f], coords_y[f]]

        if area_path[0].contains_point(parts_xy):
                staying_time[0] += 1
                if inflg[0] == 0:
                    entry[0] += 1
                    inflg[0] = 1
        else:
            inflg[0] = 0

        itr_staying[0].append(staying_time[0])
        itr_entry[0].append(entry[0])


    return entry, staying_time, itr_entry, itr_staying

