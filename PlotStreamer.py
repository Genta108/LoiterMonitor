#######################
### PlotStreamer.Py ###
#######################

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


def area_checker(coords_x, coords_y, ap, vp, video_path, area_xy, subject, loc_interval, dict_iteration, video_output):
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
