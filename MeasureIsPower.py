#########################
### MeasureIsPower.Py ###
#########################
## locomotion & velocity
## area_entry
## individual_distance

import os
import cv2
import math
import glob
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib import patches
from tqdm import tqdm

def load_gp(group):
    gp = pd.read_csv('groupprofile.csv', index_col=0)
    gp_i = gp.loc[group]
    gp_i = gp_i.values

    mp = pd.read_csv('modelprofile.csv', index_col=0)
    mp_i = mp.loc[group]
    mp_i = mp_i.values
    return gp_i, mp_i


def load_vp(vp_columns, identity, phase, experiment): #video profile loading
    vp = pd.read_csv('videoprofile.csv', index_col=0)
    if experiment == 'hdh' or experiment == 'no':
        vp_i = vp.loc[identity+'_p'+phase]
    elif experiment == 'of' or experiment == 'epm':
        vp_i = vp.loc[identity]
    elif experiment == 'gh':
        vp_i = vp.loc[identity+'_Day'+phase]
    vp_i.columns = vp_columns
    return vp_i


def load_ap(A, ap_columns, indiv, phase, experiment):
    #area profile setting
    ap = pd.read_csv('areaprofile.csv', index_col=0, skiprows=2)
    ap_d = ap.drop(ap.columns[0], axis=1)
    ap_d.columns = ap_columns
    if experiment == 'hdh' or experiment == 'no':
        ap_i = ap_d.loc[indiv+'_p'+phase]
    elif experiment == 'gh':
        ap_i = ap_d.loc[indiv+'_Day'+phase]
    elif experiment == 'of' or experiment == 'epm':
        ap_i = ap_d.loc[indiv]
    ap_i.columns = ap_columns

    area_xy = []
    area_path = []
    for i in range(A): #columns loop
        area_xy.append([])
        for j in range(0, 8, 2):
            area_xy[i].append([ap_i[i*8+j],ap_i[i*8+j+1]])
        area_path.append(path.Path(area_xy[i]))

    return ap_i, area_xy, area_path


def load_coordination(coords_columns, coords_file, csvoffset, parts):
    parts_data = []
    nose_data = []

    for i, filename in enumerate(coords_file):
        coords_data = pd.read_csv(coords_file[i], header=None, skiprows=int(csvoffset))
        coords_data.columns = coords_columns
        parts_data.append([coords_data[parts+'_x'], coords_data[parts+'_y'], coords_data[parts+'_likeli']])

    del coords_data
    gc.collect()
    return parts_data



def locomotion_calc(coords_x, coords_y, vp, param):
    fps = vp['fps']
    frame_max = int(vp['frames'])
    sc = vp['scale'] #mm/px

    loc_lim = param['loc_lim']
    loc_thre = param['loc_thre']
    loc_interval = int(param['loc_interval'])
    freeze_thre = param['freeze_thre']
    sleep_thre = param['sleep_thre']

    locomotion = []
    velocity = []
    agility = []
    active = []

    freeze_count = freeze_thre * fps #frames
    sleep_count = sleep_thre * fps #frames

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
    seq_sleeping = []

    for f in tqdm(range(0, frame_max, loc_interval), desc='progress',leave=True):
        loc_dx = 0
        loc_dy = 0
        if f == 0:
            loc_dx = coords_x[f+loc_interval]-coords_x[f]
            loc_dy = coords_y[f+loc_interval]-coords_y[f]
        else:
            loc_dx = coords_x[f]-coords_x[f-loc_interval]
            loc_dy = coords_y[f]-coords_y[f-loc_interval]

        loc_d = math.sqrt(loc_dx**2+loc_dy**2)*sc #px -> mm
        locomotion.append(loc_d)

        if loc_lim >= loc_d >= loc_thre:
            sum_locomotion += loc_d
            velocity.append(loc_d)
            agility.append(loc_d)
            freeze_count = freeze_thre * fps
            sleep_count = sleep_thre * fps
            active.append(1)
            itr_locomotion.append(sum_locomotion)
            itr_freezing.append(sum_freezing)
            itr_sleeping.append(sum_sleeping)
            seq_sleeping.append(0)
        else:
            velocity.append(0)
            active.append(0)
            freeze_count -= loc_interval
            sleep_count -= loc_interval
            if -loc_interval >= sleep_count:
                sum_sleeping += loc_interval
                freeze_count = freeze_thre * fps #frames
                seq_sleeping.append(1)
            elif 0 >= sleep_count > -loc_interval:
                sum_freezing -= sleep_thre * fps
                sum_sleeping += sleep_thre * fps + loc_interval
                seqslmax = len(seq_sleeping)
                for sl in range(1, int(sleep_thre * fps / loc_interval)+1):
                    seq_sleeping[seqslmax-sl] = 1
                seq_sleeping.append(1)
            elif 0 >= freeze_count > -loc_interval:
                sum_freezing += freeze_thre * fps + loc_interval
                seq_sleeping.append(0)
            elif -loc_interval >= freeze_count:
                sum_freezing += loc_interval #frames
                seq_sleeping.append(0)
            else:
                seq_sleeping.append(0)
            itr_locomotion.append(sum_locomotion)
            itr_freezing.append(sum_freezing)
            itr_sleeping.append(sum_sleeping)

    #velocity ave & std
    segment_max = int(frame_max/loc_interval)
    ave_velocity = sum(velocity)/segment_max
    for s in range(segment_max):
        std_velocity += (velocity[s]-ave_velocity)**2
    std_velocity = np.sqrt(std_velocity/segment_max)

    #agility ave & std
    act = sum(active) #active segment
    ave_agility = sum(agility)/act
    for a in range(act):
        std_agility += (agility[a]-ave_agility)**2
    std_agility = np.sqrt(std_agility/act)

    sum_locomotion /= 1000 #mm -> m
    sum_freezing /= fps  #frames -> seconds
    sum_sleeping /= fps

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
        'itr_sleeping':itr_sleeping,
        'seq_sleeping': seq_sleeping
    }

    return df_locomotion, dict_iteration



def entry_frames(parts_x, coords_y, area, frame_max):
    entry_stream = np.zeros(frame_max)

    #entry&staying
    for f in range(frame_max):
        parts_xy = [coords_x[f], coords_y[f]]

        if area.contains_point(parts_xy):
            entry_stream[f] = 1
        else:
            entry_stream[f] = 0

    return entry_stream



def velocity_calc(coords_x, coords_y, coords_l, vp, lh_thre):
    fps = int(vp['fps'])
    frame_max = int(vp['frames'])
    sc = vp['scale'] #mm/px

    velocity = np.zeros(frame_max)

    for f in tqdm(range(frame_max), desc='progress',leave=True):
        sum_vx = 0
        sum_vy = 0
        ef_w = 0
        if (f-fps-1) > 0 and (f+fps+1) < frame_max:
            for t in range(fps):
                if (coords_l[f-t] >= lh_thre) and (coords_l[f-t-1] >= lh_thre):
                    sum_vx = coords_x[f-t]-coords_x[f-t-1]
                    sum_vy = coords_y[f-t]-coords_y[f-t-1]
                    ef_w += 1
                if (coords_l[f+t] >= lh_thre) and (coords_l[f+t+1] >= lh_thre):
                    sum_vx = coords_x[f+t]-coords_x[f+t+1]
                    sum_vy = coords_y[f+t]-coords_y[f+t+1]
                    ef_w += 1
            if ef_w == 0:
                velocity[f] = velocity[f-1]
                continue;
            v_dx = sum_vx / ef_w
            v_dy = sum_vy / ef_w
            velocity[f] = math.sqrt(v_dx**2+v_dy**2)*sc
        elif (f-fps-1) < 0:
            v_dx = coords_x[f+1]-coords_x[f]
            v_dy = coords_y[f+1]-coords_y[f]
            velocity[f] = math.sqrt(v_dx**2+v_dy**2)*sc
        elif (f+fps+1) > frame_max:
            v_dx = coords_x[f]-coords_x[f-1]
            v_dy = coords_y[f]-coords_y[f-1]
            velocity[f] = math.sqrt(v_dx**2+v_dy**2)*sc

    return velocity


def distance_calc(pid, coords_data, setting, vp, cb):
    df_idistance = pd.DataFrame(index=[], columns=[])

    C = len(cb)
    fps = int(vp['fps'])
    frame_max = int(vp['frames'])
    sc = vp['scale'] #mm/px
    partx = 0  #setting['parts']+'_x'
    party = 1  #setting['parts']+'_y'
    likeli = 2  #setting['parts']+'_likeli'

    distance = []
    likelihood = []

    for c in range(C):
        distance.append(np.zeros(frame_max))
        likelihood.append(np.zeros(frame_max))
        for f in tqdm(range(frame_max), desc='progress',leave=True):
            distance[c][f] = math.sqrt((coords_data[cb[c][0]][partx][f]-coords_data[cb[c][1]][partx][f])**2
                                       +(coords_data[cb[c][0]][party][f]-coords_data[cb[c][1]][party][f])**2)*sc
            likelihood[c][f] = min(coords_data[cb[c][0]][likeli][f], coords_data[cb[c][1]][likeli][f])

        df_idistance['d'+pid[cb[c][0]]+'-'+pid[cb[c][1]]] = distance[c]
        df_idistance['lh'+pid[cb[c][0]]+'-'+pid[cb[c][1]]] = likelihood[c]

    return df_idistance



def list_extender(pid, distance, roommate_dlist, stranger_dlist, cb):
    C = len(cb)

    label = []
    for c in range(C):
        label.append('d'+pid[cb[c][0]]+'-'+pid[cb[c][1]])

    roommate_dlist.extend(distance[label[0]])
    stranger_dlist.extend(distance[label[1]])
    stranger_dlist.extend(distance[label[2]])
    stranger_dlist.extend(distance[label[3]])
    stranger_dlist.extend(distance[label[4]])
    roommate_dlist.extend(distance[label[5]])

    return roommate_dlist, stranger_dlist


def positioning(pid, distance, pm, vp, cb, lh_threshold, N):
    df_proximity = pd.DataFrame(index=['seconds'], columns=[])
    df_spaciality = pd.DataFrame(index=['seconds'], columns=[])
    df_cohesion = pd.DataFrame(index=['seconds'], columns=[])
    df_isolation = pd.DataFrame(index=['seconds'], columns=[])

    C = len(cb)
    fps = int(vp['fps'])
    frame_max = int(vp['frames'])
    pro_thre = pm['pro_thre']
    spa_thre = pm['spa_thre']
    pro_lim = pm['pro_lim']
    spa_lim = pm['spa_lim']

    pro_value = []
    spa_value = []
    coh_value = []
    iso_value = []

    d = []
    lh = []
    cbi = []
    for c in range(C):
        d.append('d'+pid[cb[c][0]]+'-'+pid[cb[c][1]])
        lh.append('lh'+pid[cb[c][0]]+'-'+pid[cb[c][1]])
        pro_value.append(0)
        spa_value.append(0)

    for n in range(N):
        cbi.append([])
        for c in range(C):
            if n in cb[c]:
                cbi[n].append('d'+pid[cb[c][0]]+'-'+pid[cb[c][1]])

        coh_value.append(0)
        iso_value.append(0)

    under_likeli_frame = 0
    for f in tqdm(range(frame_max), desc='progress',leave=True):
        safe_frame = 1
        for c in range(C):
            if distance[lh[c]][f] < lh_threshold:
                safe_frame = 0

        if safe_frame:
            #each combination spacing & proximity
            for c in range(C):
                if pro_lim < distance[d[c]][f] < pro_thre:
                    pro_value[c] += 1
                if spa_lim > distance[d[c]][f] > spa_thre:
                    spa_value[c] += 1

            for n in range(N):
                if spa_lim > distance[cbi[n][0]][f] > spa_thre and spa_lim > distance[cbi[n][1]][f] > spa_thre and spa_lim > distance[cbi[n][2]][f] > spa_thre:
                    iso_value[n] += 1
                if pro_lim < distance[cbi[n][0]][f] < pro_thre and pro_lim < distance[cbi[n][1]][f] < pro_thre and pro_lim < distance[cbi[n][2]][f] < pro_thre:
                    coh_value[n] += 1

        else:
            under_likeli_frame += 1

    for c in range(C):
        pro_value[c] /= fps
        spa_value[c] /= fps
        df_proximity[d[c]] = pro_value[c]
        df_spaciality[d[c]] = spa_value[c]

    for n in range(N): #unit correction
        coh_value[n] /= fps
        iso_value[n] /= fps
        df_cohesion[d[n]] = coh_value[n]
        df_isolation[d[n]] = iso_value[n]

    print(under_likeli_frame)

    return df_proximity, df_spaciality, df_cohesion, df_isolation



def interaction(pid, distance, velocity, pm, vp, cb, N):

    C = len(cb)
    fps = int(vp['fps'])
    frame_max = int(vp['frames'])
    v_thre = pm['v_thre']
    soc_entry = pm['soc_entry']
    soc_exit = pm['soc_exit']
    soc_time_thre = pm['soc_time_thre']*fps #seconds

    approach = [] #counter
    avoid = [] #counter
    soc_mode = [] #flg
    soc_count = [] #time counter
    d = [] #combination label
    directed = []
    for c in range(C):
        d.append('d'+pid[cb[c][0]]+'-'+pid[cb[c][1]])
        directed.append(pid[cb[c][0]]+'->'+pid[cb[c][1]])
        directed.append(pid[cb[c][0]]+'<-'+pid[cb[c][1]])
        approach.append([0,0])
        avoid.append([0,0])
        soc_mode.append([0,0])
        soc_count.append([0,0])

    df_approach = pd.DataFrame(index=['times'], columns=directed)
    df_avoid = pd.DataFrame(index=['times'], columns=directed)

    for f in tqdm(range(1, frame_max-1), desc='progress',leave=True):
        for c in range(C):
            if soc_entry > distance[d[c]][f]:
                if velocity[pid[cb[c][0]]][f] > v_thre and soc_mode[c][0] == 0 and soc_mode[c][1] == 0:
                    soc_mode[c][0] = 1
                    approach[c][0] += 1
                    soc_count[c][1] = 0
                elif soc_mode[c][0] == 0 and soc_mode[c][1] == 1:
                    if soc_count[c][0] >= soc_time_thre:
                        soc_mode[c][0] = 1
                        approach[c][0] += 1
                        soc_count[c][0] = 0
                    else:
                        soc_count[c][0] += 1
                if velocity[pid[cb[c][1]]][f] > v_thre and soc_mode[c][0] == 0 and soc_mode[c][1] == 0:
                    soc_mode[c][1] = 1
                    approach[c][1] += 1
                    soc_count[c][0] = 0
                elif soc_mode[c][0] == 1 and soc_mode[c][1] == 0:
                    if soc_count[c][1] >= soc_time_thre:
                        soc_mode[c][1] = 1
                        approach[c][1] += 1
                        soc_count[c][1] = 0
                    else:
                        soc_count[c][1] += 1

            elif soc_exit < distance[d[c]][f]:
                soc_count[c][0] = 0
                soc_count[c][1] = 0
                if velocity[pid[cb[c][0]]][f] > v_thre and soc_mode[c][1] == 1: #Aが動いて社会モードのBから離れた
                    soc_mode[c][1] = 0
                    avoid[c][0] += 1
                elif velocity[pid[cb[c][0]]][f] > v_thre and soc_mode[c][0] == 1: #Aが動いて社会モードを解いた
                    soc_mode[c][0] = 0
                if velocity[pid[cb[c][1]]][f] > v_thre and soc_mode[c][0] == 1: #Bが動いて社会モードのAから離れた
                    soc_mode[c][0] = 0
                    avoid[c][1] += 1
                elif velocity[pid[cb[c][1]]][f] > v_thre and soc_mode[c][1] == 1: #Bが動いて社会モードを解いた
                    soc_mode[c][1] = 0

    for c in range(C):
        df_approach[directed[c*2+0]] = approach[c][0]
        df_approach[directed[c*2+1]] = approach[c][1]
        df_avoid[directed[c*2+0]] = avoid[c][0]
        df_avoid[directed[c*2+1]] = avoid[c][1]

    return df_approach, df_avoid



def relation(pid, approach, avoid, cb, N):
    roommate = {
        pid[0]: [pid[1]],
        pid[1]: [pid[0]],
        pid[2]: [pid[3]],
        pid[3]: [pid[2]],
    }
    stranger = {
        pid[0]: [pid[2], pid[3]],
        pid[1]: [pid[2], pid[3]],
        pid[2]: [pid[0], pid[1]],
        pid[3]: [pid[0], pid[1]],
    }


    df_approach = pd.DataFrame(index=[0], columns=[])
    df_avoid = pd.DataFrame(index=[0], columns=[])
    df_beapproached = pd.DataFrame(index=[0], columns=[])
    df_beavoided = pd.DataFrame(index=[0], columns=[])
    for n in range(N):
        df_approach[pid[n]+'-roommate'+roommate[pid[n]][0]] = 0
        df_approach[pid[n]+'-stranger'+stranger[pid[n]][0]] = 0
        df_approach[pid[n]+'-stranger'+stranger[pid[n]][1]] = 0
        df_avoid[pid[n]+'-roommate'+roommate[pid[n]][0]] = 0
        df_avoid[pid[n]+'-stranger'+stranger[pid[n]][0]] = 0
        df_avoid[pid[n]+'-stranger'+stranger[pid[n]][1]] = 0
        df_beapproached[pid[n]+'-roommate'+roommate[pid[n]][0]] = 0
        df_beapproached[pid[n]+'-stranger'+stranger[pid[n]][0]] = 0
        df_beapproached[pid[n]+'-stranger'+stranger[pid[n]][1]] = 0
        df_beavoided[pid[n]+'-roommate'+roommate[pid[n]][0]] = 0
        df_beavoided[pid[n]+'-stranger'+stranger[pid[n]][0]] = 0
        df_beavoided[pid[n]+'-stranger'+stranger[pid[n]][1]] = 0

        if pid[n]+'->'+roommate[pid[n]][0] in approach.columns:
            df_approach[pid[n]+'-roommate'+roommate[pid[n]][0]] += approach[pid[n]+'->'+str(roommate[pid[n]][0])]
            df_avoid[pid[n]+'-roommate'+roommate[pid[n]][0]] += avoid[pid[n]+'->'+str(roommate[pid[n]][0])]
        else:
            df_approach[pid[n]+'-roommate'+roommate[pid[n]][0]] += approach[roommate[pid[n]][0]+'<-'+pid[n]]
            df_avoid[pid[n]+'-roommate'+roommate[pid[n]][0]] += avoid[roommate[pid[n]][0]+'<-'+pid[n]]
        if pid[n]+'->'+stranger[pid[n]][0] in approach.columns:
            df_approach[pid[n]+'-stranger'+stranger[pid[n]][0]] += approach[pid[n]+'->'+stranger[pid[n]][0]]
            df_avoid[pid[n]+'-stranger'+stranger[pid[n]][0]] += avoid[pid[n]+'->'+stranger[pid[n]][0]]
        else:
            df_approach[pid[n]+'-stranger'+stranger[pid[n]][0]] += approach[stranger[pid[n]][0]+'<-'+pid[n]]
            df_avoid[pid[n]+'-stranger'+stranger[pid[n]][0]] += avoid[stranger[pid[n]][0]+'<-'+pid[n]]
        if pid[n]+'->'+stranger[pid[n]][1] in approach.columns:
            df_approach[pid[n]+'-stranger'+stranger[pid[n]][1]] += approach[pid[n]+'->'+stranger[pid[n]][1]]
            df_avoid[pid[n]+'-stranger'+stranger[pid[n]][1]] += avoid[pid[n]+'->'+stranger[pid[n]][1]]
        else:
            df_approach[pid[n]+'-stranger'+stranger[pid[n]][1]] += approach[stranger[pid[n]][1]+'<-'+pid[n]]
            df_avoid[pid[n]+'-stranger'+stranger[pid[n]][1]] += avoid[stranger[pid[n]][1]+'<-'+pid[n]]

        if pid[n]+'<-'+roommate[pid[n]][0] in approach.columns:
            df_beapproached[pid[n]+'-roommate'+roommate[pid[n]][0]] += approach[pid[n]+'<-'+roommate[pid[n]][0]]
            df_beavoided[pid[n]+'-roommate'+roommate[pid[n]][0]] += avoid[pid[n]+'<-'+roommate[pid[n]][0]]
        else:
            df_beapproached[pid[n]+'-roommate'+roommate[pid[n]][0]] += approach[roommate[pid[n]][0]+'->'+pid[n]]
            df_beavoided[pid[n]+'-roommate'+roommate[pid[n]][0]] += avoid[roommate[pid[n]][0]+'->'+pid[n]]
        if pid[n]+'<-'+stranger[pid[n]][0] in approach.columns:
            df_beapproached[pid[n]+'-stranger'+stranger[pid[n]][0]] += approach[pid[n]+'<-'+stranger[pid[n]][0]]
            df_beavoided[pid[n]+'-stranger'+stranger[pid[n]][0]] += avoid[pid[n]+'<-'+stranger[pid[n]][0]]
        else:
            df_beapproached[pid[n]+'-stranger'+stranger[pid[n]][0]] += approach[stranger[pid[n]][0]+'->'+pid[n]]
            df_beavoided[pid[n]+'-stranger'+stranger[pid[n]][0]] += avoid[stranger[pid[n]][0]+'->'+pid[n]]
        if pid[n]+'<-'+stranger[pid[n]][1] in approach.columns:
            df_beapproached[pid[n]+'-stranger'+stranger[pid[n]][1]] += approach[pid[n]+'<-'+stranger[pid[n]][1]]
            df_beavoided[pid[n]+'-stranger'+stranger[pid[n]][1]] += avoid[pid[n]+'<-'+stranger[pid[n]][1]]
        else:
            df_beapproached[pid[n]+'-stranger'+stranger[pid[n]][1]] += approach[stranger[pid[n]][1]+'->'+pid[n]]
            df_beavoided[pid[n]+'-stranger'+stranger[pid[n]][1]] += avoid[stranger[pid[n]][1]+'->'+pid[n]]

    return df_approach, df_avoid, df_beapproached, df_beavoided



def synchronization(pid, sleep, SEG, cb, N):
    df_sync = pd.DataFrame(index=['seconds'], columns=[])

    C = len(cb)

    sync_pair = []
    sync_all = 0

    for c in range(C):
        sync_pair.append(0)

    for s in tqdm(range(SEG), desc='progress',leave=True):
        allsync = 1
        for c in range(C):
            if sleep[pid[cb[c][0]]][s] == 1 and sleep[pid[cb[c][1]]][s] == 1:
                sync_pair[c] += 1
            else:
                allsync = 0
        if allsync:
            sync_all += 1

    for c in range(C):
        df_sync[pid[cb[c][0]]+'-'+pid[cb[c][1]]] = sync_pair[c]
    df_sync['all sync.'] = sync_all

    return df_sync
