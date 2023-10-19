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
import matplotlib.style as mplstyle
mplstyle.use('fast')


def video_load(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('No video. Please check video_path.')
        print(video_path)



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



def plot_check(coords_x, coords_y, ap, vp, video_path, area_xy, subject, loc_interval, dict_iteration, video_output):
    fps = vp['fps'][0]
    frame_max = vp['frames'][0]
    active = dict_iteration['active']
    itr_locomotion = dict_iteration['itr_locomotion']
    itr_freezing = dict_iteration['itr_freezing']
    itr_sleeping = dict_iteration['itr_sleeping']

    patch = []
    for i, col in enumerate(ap.columns): #columns loop
        patch.append([])
    print('test')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('No video. Please check video_path.')
        print(video_path)

    for f in tqdm(range(frame_max), desc='progress',leave=True):
        fig = plt.figure(figsize=(12,6))
        fig.patch.set_facecolor('white')
        ax1 = fig.add_subplot(1,2,1)
        ax2 = plt.subplot(2,2,2)
        ax2.set_ylim([0, 1000])
        ax2.set_ylabel('distance (m)')
        ax3 = ax2.twinx()
        ymin,ymax = 0, 1000
        ax3.set_ylim([ymin, ymax])
        ax3.set_ylabel('time (sec.)')
        ax4 = plt.subplot(2,2,4)
        ax4.set_ylim([0, 100])
        ax4.set_ylabel('entry times')

        patch[0] = patches.Polygon(area_xy[0], closed=True, color='w', alpha=0)
        patch[1] = patches.Polygon(area_xy[1], closed=True, color='y', alpha=0.6)
        patch[2] = patches.Polygon(area_xy[2], closed=True, color='b', alpha=0.6)
        for i in range(3,len(ap.columns)):
            patch[i] = patches.Polygon(area_xy[i], closed=True, color='g', alpha=0.6)
        for i in range(len(ap.columns)):
            ax1.add_patch(patch[i])

        #ax1
        cap.set(cv2.CAP_PROP_POS_FRAMES, f) #extract a frame
        ret, frame = cap.read() #get a frame information
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #convert to color set and embeded
        if active[f]:
            text_dict = dict(boxstyle = "round", fc = 'red', ec = 'orange')
        else:
            text_dict = dict(boxstyle = "round", fc = 'gray', ec = 'green')
        ax1.annotate(subject, (coords_x[f], coords_y[f]), color="white", fontsize=8, bbox = text_dict)

        #ax2
        index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        height = [itr_locomotion[f]/1000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ax2.bar(index, height, width=0.3, tick_label=['', '', '', '', '', '', '', '', '', ''])
        ax2.annotate('{}'.format(round(height[0], 2)),
                   xy=(1, height[0]),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')

        #ax3
        height = [0, itr_freezing[f]/(fps/loc_interval), itr_sleeping[f]/(fps/loc_interval)]
        for i, col in enumerate(ap.columns): #columns loop
            height.append(dict_iteration['itr_staying_'+col][f]/fps)
        textlabel = ['move', 'freeze', 'sleep', 'all', 'cent', 'wtr', 'ul', 'ur', 'll', 'lr']
        ax3.bar(index, height, width=0.3, color='b', tick_label=textlabel)
        for n in range(1, len(index)):
            ax3.annotate('{}'.format(round(height[n], 2)),
                       xy=(index[n], height[n]),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        ax3.vlines(1.5, ymin, ymax, colors='gray', linestyle='dashed', linewidth=0.5)

        #ax4
        index = [1, 2, 3, 4, 5, 6, 7]
        height = []
        for i, col in enumerate(ap.columns): #columns loop
            height.append(dict_iteration['itr_entry_'+col][f])
        textlabel = ['allarea', 'cent', 'wtr', 'ul', 'ur', 'll', 'lr']
        ax4.bar(index, height, width=0.3, color='g', tick_label=textlabel)
        for n in range(len(index)):
            ax4.annotate('{}'.format(height[n]),
                   xy=(index[n], height[n]),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')

        #save figure
        plt.tight_layout()
        filename = str(f).zfill(5)  # 右寄せ0詰めで連番のファイル名を作成
        #plt.show()
        plt.savefig(video_output+'/'+filename+'.jpg', bbox_inches="tight", dpi=100)  # 画像保存
        plt.close()

    cv2.destroyAllWindows()
