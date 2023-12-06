#LoiterMonitor_v0.8.6
## Graph

import os
import sys
import cv2
import csv
import gc
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
import PlotStreamer as plst
import PlotGrapher as plgr

###########
# setting #
###########

args = sys.argv

experiment = args[1] #no: novel object test, epm: elevated plus maze, of: open field, tc: three chumber, gh: grouphousing, hdh: habt. dishabit.

analysis = {
    'scan_segment': 1, #seconds/segment
    'locomotion': 0,
    'synchronization': 0,
    'entry': 0,
    'velocity': 0,
    'idistance': 0,
    'histgram': 0,
    'positioning': 0,
    'interaction': 0,
    'relation': 0,
    'video': 0,
    'trajectory': 0,
    'network': 1,
}



## group housing
if experiment == 'gh':
    order = args[2]
    setting = {
        'parts': 'upperbody',
        'gp_path': '/Dropbox/B03_PopEco/Rat/rattipot/EXP-GroupHousing/'+order+'/',
        'csvoffset': '1',
    }
    if order == '1st':
        setting['fix_num'] =  '1000'
    elif order == '2nd':
        setting['fix_num'] =  '100'
    vp_columns = ['date', 'fps', 'frames', 'scale_mm/px']
    ap_columns = ['all_ulx','all_uly','all_urx','all_ury', 'all_brx', 'all_bry', 'all_blx', 'all_bly',
                  'center_ulx','center_uly','center_urx','center_ury', 'center_brx', 'center_bry', 'center_blx', 'center_bly',
                  'water_ulx','water_uly','water_urx','water_ury', 'water_brx', 'water_bry', 'water_blx', 'water_bly',
                  'cornar_ul_ulx','cornar_ul_uly','cornar_ul_urx','cornar_ul_ury', 'cornar_ul_brx', 'cornar_ul_bry', 'cornar_ul_blx', 'cornar_ul_bly',
                  'cornar_ur_ulx','cornar_ur_uly','cornar_ur_urx','cornar_ur_ury', 'cornar_ur_brx', 'cornar_ur_bry', 'cornar_ur_blx', 'cornar_ur_bly',
                  'cornar_bl_ulx','cornar_bl_uly','cornar_bl_urx','cornar_bl_ury', 'cornar_bl_brx', 'cornar_bl_bry', 'cornar_bl_blx', 'cornar_bl_bly',
                  'cornar_br_ulx','cornar_br_uly','cornar_br_urx','cornar_br_ury', 'cornar_br_brx', 'cornar_br_bry', 'cornar_br_blx', 'cornar_br_bly']
    areas = ['all', 'center', 'water dispencer','cornar ul', 'cornar ur', 'cornar bl', 'cornar br']
    N = 4
    setting['unit'] = args[3]
    setting['phase'] = args[4]
    data_path = '/Dropbox/B03_PopEco/Rat/rattipot/EXP-GroupHousing/'+order+'/'+setting['unit']+'/Day'+setting['phase']+'/'


## elevated plus maze
elif experiment == 'epm':
    order = args[2]
    setting = {
        'phase': 'NaN',
        'parts': 'upperbody',
        'gp_path': '/Dropbox/B03_PopEco/Rat/rattipot/EXP-ElevatedPlusMaze/EPM'+order+'/',
        'fix_num': '100',
        'csvoffset': '3',
        'order': order
    }
    vp_columns = ['fps', 'frames', 'start frame', 'end frame', 'scale_px/mm']
    ap_columns = ['center_ulx','center_uly','center_urx','center_ury', 'center_brx', 'center_bry', 'center_blx', 'center_bly',
                  'carmleft_ulx','carmleft_uly','carmleft_urx','carmleft_ury', 'carmleft_brx', 'carmleft_bry', 'carmleft_blx', 'carmleft_bly',
                  'carmright_ulx','carmright_uly','carmright_urx','carmright_ury', 'carmright_brx', 'carmright_bry', 'carmright_blx', 'carmright_bly',
                  'oarmup_ulx','oarmup_uly','oarmup_urx','oarmup_ury', 'oarmup_brx', 'oarmup_bry', 'oarmup_blx', 'oarmup_bly',
                  'oarmdown_ulx','oarmdown_uly','oarmdown_urx','oarmdown_ury', 'oarmdown_brx', 'oarmdown_bry', 'oarmdown_blx', 'oarmdown_bly',]
    areas = ['center', 'carmleft', 'carmright', 'oarmup', 'oarmdown']
    N = 1 #number of individual
    setting['unit'] = args[3]
    setting['lot'] = args[3][0]
    data_path = '/Dropbox/B03_PopEco/Rat/rattipot/EXP-ElevatedPlusMaze/EPM'+order+'/'+setting['lot']+'/'



## novel object
elif experiment == 'no':
    order = args[2]
    setting = {
        'parts': 'upperbody',
        'gp_path': '/Dropbox/B03_PopEco/Rat/rattipot/EXP-NovelObject/NO'+order+'/',
        'fix_num': '100',
        'csvoffset': '3',
        'order': order
    }
    vp_columns = ['fps', 'frames', 'start frame', 'end frame', 'scale_px/mm']
    ap_columns = ['all_ulx','all_uly','all_urx','all_ury', 'all_brx', 'all_bry', 'all_blx', 'all_bly',
                  'wide_ulx','wide_uly','wide_urx','wide_ury', 'wide_brx', 'wide_bry', 'wide_blx', 'wide_bly',
                  'center_ulx','center_uly','center_urx','center_ury', 'center_brx', 'center_bry', 'center_blx', 'center_bly',
                  'objnose_ulx','objnose_uly','objnose_urx','objnose_ury', 'objnose_brx', 'objnose_bry', 'objnose_blx', 'objnose_bly',
                  'objhead_ulx','objhead_uly','objhead_urx','objhead_ury', 'objhead_brx', 'objhead_bry', 'objhead_blx', 'objhead_bly',
                  'objneck_ulx','objneck_uly','objneck_urx','objneck_ury', 'objneck_brx', 'objneck_bry', 'objneck_blx', 'objneck_bly']
    areas = ['all', 'wide', 'center', 'objnose', 'objhead', 'objneck']
    N = 1 #number of individual
    setting['unit'] = args[3]
    setting['lot'] = args[3][0]
    setting['phase'] = args[4]
    data_path = '/Dropbox/B03_PopEco/Rat/rattipot/EXP-NovelObject/NO'+order+'/'+setting['lot']+'/'



## open field
elif experiment == 'of':
    order = args[2]
    setting = {
        'phase': 'NaN',
        'parts': 'upperbody',
        'gp_path': '/Dropbox/B03_PopEco/Rat/rattipot/EXP-OpenField/OF'+order+'/',
        'fix_num': '100',
        'csvoffset': '3',
        'order': order
    }
    vp_columns = ['fps', 'frames', 'start frame', 'end frame', 'scale_px/mm']
    ap_columns = ['all_ulx','all_uly','all_urx','all_ury', 'all_brx', 'all_bry', 'all_blx', 'all_bly',
                  'wide_ulx','wide_uly','wide_urx','wide_ury', 'wide_brx', 'wide_bry', 'wide_blx', 'wide_bly',
                  'center_ulx','center_uly','center_urx','center_ury', 'center_brx', 'center_bry', 'center_blx', 'center_bly']
    areas = ['all', 'wide', 'center']
    N = 1 #number of individual
    setting['unit'] = args[3]
    setting['lot'] = args[3][0]
    data_path = '/Dropbox/B03_PopEco/Rat/rattipot/EXP-OpenField/OF'+order+'/'+setting['lot']+'/'



## habituation dishabituation
elif experiment == 'hdh':
    setting = {
        'parts': 'upperbody',
        'gp_path': '/Dropbox/B03_PopEco/Rat/rattipot/EXP-HabitDishabit/',
        'fix_num': '100',
        'csvoffset': '3'
    }
    vp_columns = ['fps', 'frames', 'start frame', 'end frame', 'scale_px/mm']
    ap_columns = ['all_ulx','all_uly','all_urx','all_ury', 'all_brx', 'all_bry', 'all_blx', 'all_bly',
                  'half_ulx','half_uly','half_urx','half_ury', 'half_brx', 'half_bry', 'half_blx', 'half_bly',
                  'front_ulx','front_uly','front_urx','front_ury', 'front_brx', 'front_bry', 'front_blx', 'front_bly',
                  'wfront_ulx','wfront_uly','wfront_urx','wfront_ury', 'wfront_brx', 'wfront_bry', 'wfront_blx', 'wfront_bly',
                  'lc_ulx','lc_uly','lc_urx','lc_ury', 'lc_brx', 'lc_bry', 'lc_blx', 'lc_bly',
                  'rc_ulx','rc_uly','rc_urx','rc_ury', 'rc_brx', 'rc_bry', 'rc_blx', 'rc_bly',
                  'nose_ulx','nose_uly','nose_urx','nose_ury', 'nose_brx', 'nose_bry', 'nose_blx', 'nose_bly']
    areas = ['all', 'half', 'front', 'wfront','lc', 'rc', 'nose']
    area_data = ['all', 'half', 'front', 'wfront','lc', 'rc', 'nose', 'nose_wfront', 'nose_cornar']
    AD = len(area_data)
    N = 1 #number of individual
    setting['lot'] = args[2]
    setting['unit'] = args[3]
    setting['phase'] = args[4]
    data_path = '/Dropbox/B03_PopEco/Rat/rattipot/EXP-HabitDishabit/iteration3_'+setting['lot']



A = len(areas) #number of areaprofile columns


if analysis['video'] or analysis['trajectory']:
    if experiment == 'gh':
        video_path = glob.glob('/Volumes/Extreme SSD/rat_society/hakataya2023/'+setting['unit']+'_Day'+setting['phase']+'*.mp4')
        #video_path = glob.glob('/mnt/data_complex/syncbox/rat_analysis/'+setting_dict['group']+'/'+setting_dict['day']+'/'+setting_dict['group']+'*.mp4')
        video_output = 'video/'
        #video_output = '/mnt/data_complex/syncbox/rat_analysis/'+setting_dict['group']+'/'+setting_dict['day']+'/Video/'
        dist_output = ''
    elif experiment == 'epm':
        video_path = glob.glob('/Volumes/Extreme SSD/rat_society/rat_character2023/EPM'+order+'/crop&gray/EPM'+order+'_'+setting['unit']+'_GC.mp4')
        dist_output = setting['lot']
    elif experiment == 'no':
        video_path = glob.glob('/Volumes/Extreme SSD/rat_society/rat_character2023/NO'+order+'/crop&gray/NO'+order+'_'+setting['unit']+'_p'+setting['phase']+'_GC.mp4')
        dist_output = setting['lot']
    elif experiment == 'of':
        video_path = glob.glob('/Volumes/Extreme SSD/rat_society/rat_character2023/OF'+order+'/crop&gray/OF'+order+'_'+setting['unit']+'_GC.mp4')
        dist_output = setting['lot']
    elif experiment == 'hdh':
        video_path = glob.glob('/Volumes/Extreme SSD/rat_society/HDH2023/crop/HDH_'+setting['unit']+'_p'+setting['phase']+'_C.mp4')
        video_output = 'iteration3_'+setting['lot']+'/videotest'
        dist_output = 'iteration3_'+setting['lot']

    print(video_path)



###########
# default #
###########

home = os.path.expanduser("~")

os.chdir(home+setting['gp_path'])
vp = mip.load_vp(vp_columns, setting['unit'], setting['phase'], experiment)
fps = vp['fps']
frame_max = int(vp['frames'])
start_frame = vp['start of frame']
end_frame = vp['end of frame']
sc = vp['scale'] # mm/px

pid = setting['unit']
if experiment == 'gh':
    pid, mid = mip.load_gp(setting['unit'])

coords_columns=['frame', 'nose_x', 'nose_y', 'nose_likeli', 'head_x', 'head_y', 'head_likeli',
                'earR_x', 'earR_y', 'earR_likeli', 'earL_x', 'earL_y', 'earL_likeli',
                'neck_x', 'neck_y', 'neck_likeli', 'upperbody_x', 'upperbody_y','upperbody_likeli',
                'lowerbody_x', 'lowerbody_y','lowerbody_likeli', 'tailbase_x', 'tailbase_y','tailbase_likeli']

lh_thre = 0.8 #likelihood threshold

locomotion_pm = {
    'loc_lim': 1000,    #locomotion (mm) per scan segment
    'loc_thre': 10,      #mm per scan segment, 1st: 10, 2nd:10
    'freeze_thre': 5,  #seconds
    'sleep_thre': 100   #seconds
}

position_pm = {
    'spa_thre': 400, #mm
    'spa_lim': 1200, #mm
    'pro_thre': 200, #mm
    'pro_lim': 0,
}

interaction_pm = {
    'v_thre': 0.67, #mm/frame, 1st:0.67, 2nd:0.42
    'soc_entry': 150,
    'soc_exit': 300,
    'soc_time_thre': 3, #seconds
}

#combination
if N == 2:
    cb = [[0, 1]]
    pm = [[0, 1], [1, 0]]
elif N == 3:
    cb = [[0, 1], [0, 2], [1, 2]]
    pm = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]
elif N == 4:
    cb = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    pm = [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]]
C = len(cb)
P = len(pm)

#time segmentation in locomotion
if analysis['scan_segment']:
    locomotion_pm['loc_interval'] = analysis['scan_segment']*fps #frames
else:
    locomotion_pm['loc_interval'] = 1 #frames



################
# main process #
################
os.chdir(home+data_path)

#coordination & locomotion file setting
if experiment == 'gh':
    coords_file = []
    for n in range(N):
        coords_file.append(glob.glob(setting['unit']+'_Day'+setting['phase']+'*DLC_resnet50_*'+mid[n]+'*fixed'+setting['fix_num']+'.csv')[0])
elif experiment == 'no':
    coords_file = sorted(glob.glob('NO'+order+'_'+setting['unit']+'_p'+setting['phase']+'*DLC*.csv'))
elif experiment == 'epm':
    coords_file = sorted(glob.glob('EPM'+order+'_'+setting['unit']+'*DLC*.csv'))
elif experiment == 'of':
    coords_file = sorted(glob.glob('OF'+order+'_'+setting['unit']+'*DLC*.csv'))
elif experiment == 'hdh':
    coords_file = sorted(glob.glob('HDH_'+setting['unit']+'_p'+setting['phase']+'*DLC_resnet50_*300000.csv'))


parts_data = mip.load_coordination(coords_columns, coords_file, setting['csvoffset'], setting['parts'])

if experiment == 'no':
    nose_data = mip.load_coordination(coords_columns, coords_file, setting['csvoffset'], 'nose')
    head_data = mip.load_coordination(coords_columns, coords_file, setting['csvoffset'], 'head')
    neck_data = mip.load_coordination(coords_columns, coords_file, setting['csvoffset'], 'neck')
elif experiment == 'hdh':
    nose_data = mip.load_coordination(coords_columns, coords_file, setting['csvoffset'], 'nose')



#######################
##### locomotion ######
#######################
if analysis['locomotion']:
    os.chdir(home+data_path)
    df_sleeping = pd.DataFrame()

    for n in range(N):
        df_locomotion, locomo_iteration = mip.locomotion_calc(parts_data[n][0], parts_data[n][1], vp, locomotion_pm) #locomotion data calcuration

        print(df_locomotion)

        if experiment == 'gh':
            df_locomotion.to_csv(pid[n]+'_Day'+setting['phase']+'_locomotion>'+str(locomotion_pm['loc_thre'])+'mmseg='+str(analysis['scan_segment'])+'sec_'+setting['parts']+'.csv')
            df_sleeping[pid[n]] = locomo_iteration['seq_sleeping']
        elif experiment == 'no':
            df_locomotion.to_csv('NO'+order+'_'+setting['unit']+'_p'+setting['phase']+'_locomotion_'+setting['parts']+'.csv')
        elif experiment == 'of':
            df_locomotion.to_csv('OF'+order+'_'+setting['unit']+'_locomotion_'+setting['parts']+'.csv')
        elif experiment == 'epm':
            df_locomotion.to_csv('EPM'+order+'_'+setting['unit']+'_locomotion_'+setting['parts']+'.csv')
        elif experiment == 'hdh':
            df_locomotion.to_csv('HDH_'+setting['unit']+'_p'+setting['phase']+'_locomotion_'+setting['parts']+'.csv')

        if analysis['trajectory']:
            os.chdir(home+setting['gp_path'])
            ap, area_xy, area_path = mip.load_ap(A, ap_columns, setting['unit'], setting['phase'], experiment)
            os.chdir(home+data_path)
            plgr.trajectory(pid[n], parts_data[n], nose_data[n], areas, ap, vp, video_path[0], area_xy, setting, dist_output, experiment)

        if analysis['video']:
            plst.movie_checker(parts_data[n], areas, ap, vp, video_path[n], area_xy, setting['unit'], locomotion_pm['loc_interval'], dict_iteration, video_output)

    if experiment == 'gh':
        df_sleeping.to_csv(setting['unit']+'_Day'+setting['phase']+'_sleeping_'+setting['parts']+'.csv')
    del df_locomotion
    gc.collect()


###########################
##### synchronization #####
###########################
if analysis['synchronization'] and experiment == 'gh':
    print('---- sync summation ----> '+setting['unit']+'_Day'+setting['phase'])
    os.chdir(home+data_path)
    sleep_file = setting['unit']+'_Day'+setting['phase']+'_sleeping_'+setting['parts']+'.csv'
    sleep = pd.read_csv(sleep_file, index_col = 0)

    df_sync = mip.synchronization(pid, sleep, int(frame_max / locomotion_pm['loc_interval']),cb, N)
    df_sync.to_csv(setting['unit']+'_Day'+setting['phase']+'_synchronization_'+setting['parts']+'.csv')

    del df_sync
    gc.collect()



##################
##### entry ######
##################
if analysis['entry']:
    print('---- entry calculation ----> '+setting['unit']+'_Day'+setting['phase'])
    for n in range(N):
        os.chdir(home+setting['gp_path'])
        entry_iteration = pd.DataFrame()
        ap, area_xy, area_path = mip.load_ap(A, ap_columns, setting['unit'], setting['phase'], experiment)

        os.chdir(home+data_path)

        df_entry = pd.DataFrame(index=[0], columns=[])
        entry_body = []
        entry = np.zeros(A)
        itr_entry = []
        staying_time = np.zeros(A)
        itr_staying = []

        for a in range(A):
            entry_body.append(mip.entry_frames(parts_data[n][0], parts_data[n][1], area_path[a], frame_max)) #entry&staying area calcuration
        if experiment == 'no':
            entry_nose = mip.entry_frames(nose_data[n][0], nose_data[n][1], area_path[3], frame_max) #entry&staying area calcuration
            entry_head = mip.entry_frames(head_data[n][0], head_data[n][1], area_path[4], frame_max) #entry&staying area calcuration
            entry_neck = mip.entry_frames(neck_data[n][0], neck_data[n][1], area_path[5], frame_max) #entry&staying area calcuration
        if experiment == 'hdh':
            entry_nose = mip.entry_frames(nose_data[n][0], nose_data[n][1], area_path[6], frame_max) #entry&staying area calcuration
            entry = np.zeros(AD)
            staying_time = np.zeros(AD)

        ###entry times
        for a, col in enumerate(areas):
            itr_entry.append([])
            for f in range(frame_max):
                if f == frame_max-1:
                    entry[a] += 0
                elif entry_body[a][f] == 0 and entry_body[a][f+1] == 1:
                    entry[a] += 1
                itr_entry[a].append(entry[a])
            df_entry['entry to '+col] = entry[a]
            entry_iteration['itr_entry_'+col] = itr_entry[a]

        #staying exception
        if experiment == 'no':
            itr_entry = []

            itr_entry.append([])
            entry[A-3] = 0 #object nose
            for f in range(frame_max):
                if f == frame_max-1:
                    entry[A-3] += 0
                elif entry_nose[f] == 0 and entry_nose[f+1] == 1:
                    entry[A-3] += 1
                itr_entry[0].append(entry[A-3])
            df_entry['entry to '+areas[A-3]] = entry[A-3]
            entry_iteration['itr_entry_'+areas[A-3]] = itr_entry[0]

            itr_entry.append([])
            entry[A-2] = 0 #object head
            for f in range(frame_max):
                if f == frame_max-1:
                    entry[A-2] += 0
                elif entry_head[f] == 0 and entry_head[f+1] == 1:
                    entry[A-2] += 1
                itr_entry[1].append(entry[A-2])
            df_entry['entry to '+areas[A-2]] = entry[A-2]
            entry_iteration['itr_entry_'+areas[A-2]] = itr_entry[1]

            itr_entry.append([])
            entry[A-1] = 0
            for f in range(frame_max):
                if f == frame_max-1:
                    entry[A-1] += 0
                elif entry_neck[f] == 0 and entry_neck[f+1] == 1:
                    entry[A-1] += 1
                itr_entry[2].append(entry[A-1])
            df_entry['entry to '+areas[A-1]] = entry[A-1]
            entry_iteration['itr_entry_'+areas[A-1]] = itr_entry[2]

        if experiment == 'hdh':
            itr_entry = []

            itr_entry.append([])
            entry[AD-3] = 0 #nose
            for f in range(frame_max):
                if f == frame_max-1:
                    entry[AD-3] += 0
                elif entry_nose[f] == 0 and entry_nose[f+1] == 1:
                    entry[AD-3] += 1
                itr_entry[0].append(entry[AD-3])
            entry_iteration['itr_entry'+area_data[AD-3]] = itr_entry[0]

            itr_entry.append([])
            entry[AD-2] = 0 #nose front
            for f in range(frame_max):
                if f == frame_max-1:
                    entry[AD-2] += 0
                elif entry_nose[f] == 0 and entry_nose[f+1] == 1 and entry_body[2][f] == 1:
                    entry[AD-2] += 1
                itr_entry[1].append(entry[AD-2])
            entry_iteration['itr_entry'+area_data[AD-2]] = itr_entry[1]

            itr_entry.append([])
            entry[AD-1] = 0 #nose cornar
            for f in range(frame_max):
                if f == frame_max-1:
                    entry[AD-1] += 0
                elif entry_nose[f] == 0 and entry_nose[f+1] == 1 and (entry_body[4][f] == 1 or entry_body[5][f] == 1):
                    entry[AD-1] += 1
                itr_entry[2].append(entry[AD-1])
            entry_iteration['itr_entry'+area_data[AD-1]] = itr_entry[2]

        ###staying time
        for a, col in enumerate(areas):
            itr_staying.append([])
            for f in range(frame_max):
                staying_time[a] += entry_body[a][f]
                itr_staying[a].append(staying_time[a])
            staying_time[a] /= fps #convert frames -> seconds
            df_entry['time in '+col+'(sec.)'] = staying_time[a]
            entry_iteration['itr_staying_'+col] = itr_staying[a]

        #staying exception
        if experiment == 'no':
            itr_staying = []

            itr_staying.append([])
            for f in range(frame_max):
                staying_time[A-3] += entry_nose[f]
                itr_staying[0].append(staying_time[A-3])
            staying_time[A-3] /= fps
            df_entry['time in '+areas[0]+'(sec.)'] = staying_time[A-3]
            entry_iteration['itr_staying_'+areas[A-3]] = itr_staying[0]

            itr_staying.append([])
            for f in range(frame_max):
                staying_time[A-2] += entry_head[f]
                itr_staying[1].append(staying_time[A-2])
            staying_time[A-2] /= fps
            df_entry['time in '+areas[A-2]+'(sec.)'] = staying_time[A-2]
            entry_iteration['itr_staying_'+areas[A-2]] = itr_staying[1]

            itr_staying.append([])
            for f in range(frame_max):
                staying_time[A-1] += entry_neck[f]
                itr_staying[2].append(staying_time[A-1])
            staying_time[A-1] /= fps
            df_entry['time in '+areas[A-1]+'(sec.)'] = staying_time[A-1]
            entry_iteration['itr_staying_'+areas[A-1]] = itr_staying[2]

        if experiment == 'hdh':
            itr_staying = []

            itr_staying.append([])
            for f in range(frame_max):
                staying_time[AD-3] += entry_nose[f]
                itr_staying[0].append(staying_time[AD-3])
            staying_time[AD-3] /= fps
            df_entry['time in '+area_data[AD-3]+'(sec.)'] = staying_time[AD-3]
            entry_iteration['itr_staying_'+area_data[AD-3]] = itr_staying[0]

            itr_staying.append([])
            for f in range(frame_max):
                if entry_body[3][f]:
                    staying_time[AD-2] += entry_nose[f]
                itr_staying[1].append(staying_time[AD-2])
            staying_time[AD-2] /= fps
            df_entry['time in '+area_data[AD-2]+'(sec.)'] = staying_time[AD-2]
            entry_iteration['itr_staying_'+area_data[AD-2]] = itr_staying[1]

            itr_staying.append([])
            for f in range(frame_max):
                if entry_body[4][f] == 1 or entry_body[5][f] == 1:
                    staying_time[AD-1] += entry_nose[f]
                itr_staying[2].append(staying_time[AD-1])
            staying_time[AD-1] /= fps
            df_entry['time in '+area_data[AD-1]+'(sec.)'] = staying_time[AD-1]
            entry_iteration['itr_staying_'+area_data[AD-1]] = itr_staying[2]

        print(df_entry)
        if experiment == 'gh':
            df_entry.to_csv(pid[n]+'_Day'+setting['phase']+'_entry_'+setting['parts']+'.csv')
        elif experiment == 'no':
            df_entry.to_csv('NO'+order+'_'+setting['unit']+'_p'+setting['phase']+'_entry_'+setting['parts']+'.csv')
        elif experiment == 'of':
            df_entry.to_csv('OF'+order+'_'+setting['unit']+'_entry_'+setting['parts']+'.csv')
        elif experiment == 'epm':
            df_entry.to_csv('EPM'+order+'_'+setting['unit']+'_entry_'+setting['parts']+'.csv')
        elif experiment == 'hdh':
            df_entry.to_csv('HDH_'+setting['unit']+'_p'+setting['phase']+'_entry_'+setting['parts']+'.csv')

    del df_entry
    gc.collect()



####################
##### velocity #####
####################
if analysis['velocity']:
    df_velocity = pd.DataFrame(index=[], columns=[])

    if experiment == 'hdh':
        for n in range(N):
            ap, area_xy, area_path = mip.load_ap(A, ap_columns, setting['unit'], setting['phase'], experiment)
            df_velocity['subject'+str(n)] = mip.velocity_calc(parts_data[n][0], parts_data[n][1], parts_data[n][2], vp, lh_thre)  #velocity data calcuration
            df_velocity.to_csv('HDH_'+setting['unit']+'_p'+setting['phase']+'_velocity_'+setting['parts']+'.csv')

    if experiment == 'gh':
        os.chdir(home+data_path)
        print('---- velocity calculation ----> '+setting['unit']+'_Day'+setting['phase'])
        for n in range(N):
            df_velocity[pid[n]] = mip.velocity_calc(parts_data[n][0], parts_data[n][1], parts_data[n][2], vp, lh_thre)  #velocity data calcuration
        df_velocity.to_csv(setting['unit']+'_Day'+setting['phase']+'_velocity_'+setting['parts']+'.csv')

    del df_velocity
    gc.collect()



###########################
##### indiv. distance #####
###########################
if analysis['idistance'] and experiment == 'gh':
    print('---- distance calculation ----> '+setting['unit']+'_Day'+setting['phase'])
    os.chdir(home+data_path)
    df_idistance = mip.distance_calc(pid, parts_data, setting, vp, cb)  #distance data calcuration
    df_idistance.to_csv(setting['unit']+'_Day'+setting['phase']+'_distance_'+setting['parts']+'.csv')

    del df_idistance
    gc.collect()


del parts_data
if experiment == 'no':
    del nose_data
    del head_data
    del neck_data
if experiment == 'hdh':
    del nose_data
gc.collect()



#############################
##### distance histgram #####
#############################
if analysis['histgram'] and experiment == 'gh':
    print('---- histgram generation ----> '+setting['unit']+'_Day'+setting['phase'])

    if setting['unit'] == 'AllGroup':
        os.chdir(home+setting['gp_path']+'AllGroup/')
        roommate_file = 'AllGroup_Day'+setting['phase']+'_roommate-dlist_'+setting['parts']+'.csv'
        roommate_distance = pd.read_csv(roommate_file)
        plgr.histgram(roommate_distance, 'roommate distance', setting)

        del roommate_dlist

        stranger_file = 'AllGroup_Day'+setting['phase']+'_stranger-dlist_'+setting['parts']+'.csv'
        stranger_distance = pd.read_csv(stranger_file, 'stranger distance', setting)
        plgr.histgram(stranger_dlist)

        del stranger_dlist

    else:
        os.chdir(home+data_path)
        distance_file = setting['unit']+'_Day'+setting['phase']+'_roommate_distance_'+setting['parts']+'.csv'
        distance = pd.read_csv(distance_file)





#######################
##### positioning #####
#######################
if analysis['positioning'] and experiment == 'gh':
    print('---- positioning calculation ----> '+setting['unit']+'_Day'+setting['phase'])

    os.chdir(home+data_path)

    distance = pd.read_csv(distance_file)

    df_proximity, df_spaciality, df_cohesion, df_isolation = mip.positioning(pid, distance, position_pm, vp, cb, lh_thre, N)
    df_proximity.to_csv(setting['unit']+'_Day'+setting['phase']+'_proximity_'+setting['parts']+'.csv')
    df_spaciality.to_csv(setting['unit']+'_Day'+setting['phase']+'_spaciality_'+setting['parts']+'.csv')
    df_cohesion.to_csv(setting['unit']+'_Day'+setting['phase']+'_cohesion_'+setting['parts']+'.csv')
    df_isolation.to_csv(setting['unit']+'_Day'+setting['phase']+'_isolation_'+setting['parts']+'.csv')

    del df_proximity
    del df_spaciality
    del df_cohesion
    del df_isolation
    gc.collect()



#######################
##### interaction #####
#######################
if analysis['interaction'] and experiment == 'gh':
    print('---- interaction calculation ----> '+setting['group']+'_Day'+setting['day'])

    os.chdir(home+data_path)
    distance_file = setting['group']+'_Day'+setting['day']+'_distance_'+setting['parts']+'.csv'
    distance = pd.read_csv(distance_file)
    velocity_file = setting['group']+'_Day'+setting['day']+'_velocity_'+setting['parts']+'.csv'
    velocity = pd.read_csv(velocity_file)

    df_approach, df_avoid = mip.interaction(pid, distance, velocity, interaction_pm, vp, cb, N)
    df_approach.to_csv(setting['group']+'_Day'+setting['day']+'_approach_'+setting['parts']+'.csv')
    df_avoid.to_csv(setting['group']+'_Day'+setting['day']+'_avoid_'+setting['parts']+'.csv')

    del df_approach
    del df_avoid
    gc.collect()



################################
##### relation arrangement #####
################################
if analysis['relation'] and experiment == 'gh':
    print('---- relation summation ----> '+setting['group']+'_Day'+setting['day'])

    os.chdir(home+data_path)
    approach_file = setting['group']+'_Day'+setting['day']+'_approach_'+setting['parts']+'.csv'
    approach = pd.read_csv(approach_file)
    avoid_file = setting['group']+'_Day'+setting['day']+'_avoid_'+setting['parts']+'.csv'
    avoid = pd.read_csv(avoid_file)

    df_relatedapproach, df_relatedavoid, df_relatedbeapproached, df_relatedbeavoided = mip.relation(pid, approach, avoid, cb, N)
    df_relatedapproach.to_csv(setting['group']+'_Day'+setting['day']+'_relatedapproach_'+setting['parts']+'.csv')
    df_relatedavoid.to_csv(setting['group']+'_Day'+setting['day']+'_relatedavoid_'+setting['parts']+'.csv')
    df_relatedbeapproached.to_csv(setting['group']+'_Day'+setting['day']+'_relatedbeapproached_'+setting['parts']+'.csv')
    df_relatedbeavoided.to_csv(setting['group']+'_Day'+setting['day']+'_relatedbeavoided_'+setting['parts']+'.csv')

    del df_relatedapproach
    del df_relatedavoid
    del df_relatedbeapproached
    del df_relatedbeavoided
    gc.collect()



###################
##### network #####
###################
if analysis['network'] and experiment == 'gh':
    print('---- network drawing ----> '+setting['unit']+'_Day'+setting['phase'])
    os.chdir(home+data_path)


    dependant = ['proximity', 'spaciality']
    for i, var in enumerate(dependant):
        title = var+' (sec.)'
        if setting['unit'] == 'AllGroup':
            filename = setting['unit']+'_Day'+setting['phase']+'_'+var+'_'+setting['parts']+'.csv'
        else:
            filename = setting['unit']+'_Day'+setting['phase']+'_'+var+'_'+setting['parts']+'.csv'
            data = pd.read_csv(filename, index_col = 0)
            w_coef = min(data.loc['seconds'])/10
        plgr.graphnet(pid, cb, data, w_coef, title, setting)


    dependant = ['approach', 'avoid', 'beapproached', 'beavoided']
    for i, var in enumerate(dependant):
        title = var+' (times)'
        if setting['unit'] == 'AllGroup':
            filename = setting['unit']+'_Day'+setting['phase']+'_related'+var+'_'+setting['parts']+'.csv'
        else:
            filename = setting['unit']+'_Day'+setting['phase']+'_related'+var+'_'+setting['parts']+'.csv'
            data = pd.read_csv(filename, index_col = 0)
            std_coef = sum(data.loc[0])/P
        plgr.direct_graphnet(pid, cb, pm, data, std_coef, title, setting)





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
