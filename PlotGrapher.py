#######################
### PlotStreamer.Py ###
#######################

import cv2
import math
import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib import patches
from tqdm import tqdm
import matplotlib.style as mplstyle



def trajectory(pid, parts_data, nose_data, areas, ap, vp, video_path, area_xy, setting, dist_output, experiment):
    coords_x = parts_data[0]
    coords_y = parts_data[1]
    nose_x = nose_data[0]
    nose_y = nose_data[1]
    fps = vp['fps']
    frame_max = int(vp['frames'])
    unit = setting['unit']
    phase = setting['phase']

    patch = []
    for i, col in enumerate(areas): #columns loop
        patch.append([])
    print('test')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('No video. Please check video_path.')
        print(video_path)

    fig = plt.figure(figsize=(8,8))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot()

    if experiment == 'epm':
        patch[0] = patches.Polygon(area_xy[0], closed=True, color='w', alpha=0.4)
        patch[1] = patches.Polygon(area_xy[1], closed=True, color='g', alpha=0.4)
        patch[2] = patches.Polygon(area_xy[2], closed=True, color='g', alpha=0.4)
        patch[3] = patches.Polygon(area_xy[3], closed=True, color='y', alpha=0.4)
        patch[4] = patches.Polygon(area_xy[4], closed=True, color='y', alpha=0.4)
    elif experiment == 'no':
        patch[0] = patches.Polygon(area_xy[1], closed=True, color='w', alpha=0.1)
        patch[1] = patches.Polygon(area_xy[0], closed=True, color='w', alpha=0.2)
        patch[2] = patches.Polygon(area_xy[2], closed=True, color='g', alpha=0.3)
        patch[3] = patches.Polygon(area_xy[5], closed=True, color='b', alpha=0.4)
        patch[4] = patches.Polygon(area_xy[4], closed=True, color='y', alpha=0.4)
        patch[5] = patches.Polygon(area_xy[3], closed=True, color='r', alpha=0.4)
    #patch[6] = patches.Polygon(area_xy[6], closed=True, color='r', alpha=0.6)
    #patch[7] = patches.Polygon(area_xy[7], closed=True, color='r', alpha=0.6)
    for i in range(len(areas)):
        ax.add_patch(patch[i])

    #ax1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #extract a frame
    ret, frame = cap.read() #get a frame information
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #convert to color set and embeded
    text_dict = dict(boxstyle = "round", fc = 'gray', ec = 'green')
    if experiment == 'gh':
        ax.annotate(pid, (coords_x[0], coords_y[0]), color="white", fontsize=8, bbox = text_dict)
        ax.plot(coords_x, coords_y, color = 'orange', marker='.', linewidth=2, alpha=0.01)
    elif experiment == 'no' or experiment == 'of' or experiment == 'epm':
        ax.plot(coords_x, coords_y, color = 'orange', marker='.', linewidth=2, alpha=0.5)
    elif experiment == 'hdh':
        ax.annotate(subject, (coords_x[0], coords_y[0]), color="white", fontsize=8, bbox = text_dict)
        ax.plot(nose_x, nose_y, color = 'orange', marker='.', linewidth=2, alpha=0.5)

    #save figure
    ax.axis("off")
    plt.tight_layout()
    #plt.show()
    if experiment == 'hdh':
        plt.savefig(dist_output+'/HDH_'+unit+'_p'+phase+'_trajectory.jpg', bbox_inches="tight", dpi=180)  # 画像保存
    elif experiment == 'no':
        plt.savefig('NO'+setting['order']+'_'+unit+'_p'+phase+'_trajectory.jpg', bbox_inches="tight", dpi=180)  # 画像保存
    elif experiment == 'of':
        plt.savefig('OF'+setting['order']+'_'+unit+'_trajectory.jpg', bbox_inches="tight", dpi=180)  # 画像保存
    elif experiment == 'epm':
        plt.savefig('EPM'+setting['order']+'_'+unit+'_trajectory.jpg', bbox_inches="tight", dpi=180)  # 画像保存
    elif experiment =='gh':
        plt.savefig(pid+'_Day'+setting['phase']+'_'+setting['parts']+'_trajectory.jpg', bbox_inches="tight", dpi=180)  # 画像保存
    plt.close()

    cv2.destroyAllWindows()


def histgram(dlist, title, setting):
    mpl.rcParams['axes.ymargin'] = 0.5
    mpl.rcParams['axes.xmargin'] = 0.5
    plt.rcParams['figure.subplot.left'] = 0.15
    plt.rcParams['figure.subplot.bottom'] = 0.15

    fig = plt.figure()
    ax = plt.subplot()
    ax.hist(dlist, 50)
    plt.title(title, fontsize=18)
    plt.show()



def graphnet(pid, cb, matrix, w_coef, title, setting):
    N = len(pid)
    C = len(cb)

    #Distance Graph object
    G = nx.Graph()
    for n in range(N):
        G.add_node(pid[n])

    nvalue = []
    data_label = []
    edge_weights = []
    for c in range(C):
        G.add_edge(pid[cb[c][0]], pid[cb[c][1]])
        data_label.append('d'+pid[cb[c][0]]+'-'+pid[cb[c][1]])
        nvalue.append(round(matrix[data_label[c]].to_list()[0]))

        edge_weights.append(nvalue[c]/w_coef) #weight of edge

        G[pid[cb[c][0]]][pid[cb[c][1]]]["weight"] = nvalue[c]

    pos = {}
    if N == 4:
        pos[pid[0]] = (0,1)
        pos[pid[1]] = (0,0)
        pos[pid[2]] = (1,1)
        pos[pid[3]] = (1,0)

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot()
    plt.title(title, fontsize=16)

    edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, label_pos=0.65)
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=['blue', 'red', 'gray', 'orange'])
    nx.draw_networkx_labels(G, pos, font_size=20, font_color='w')
    nx.draw(G, pos, ax, with_labels=True, font_size=20, node_size=1500, node_color=['blue', 'red', 'gray', 'orange'], edge_color='purple', font_color='w', width=edge_weights, alpha=0.7)
    #plt.show()
    plt.savefig(setting['unit']+'_Day'+setting['phase']+'_'+title+'_network.jpg', bbox_inches="tight", dpi=180)  # 画像保存



def direct_graphnet(pid, cb, pm,  matrix, std_coef, title, setting):
    N = len(pid)
    C = len(cb)
    P = len(pm)

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

    #Directed Graph object
    DG = []

    ncolor = ['blue', 'red', 'gray', 'orange']

    DGN = nx.DiGraph()

    data_label = []
    for n in range(N):
        DG.append([])
        DGN.add_node(pid[n])
        DG[n] = nx.DiGraph()
        DG[n].add_node(pid[n])
        data_label.append(pid[n]+'-roommate'+roommate[pid[n]][0])
        data_label.append(pid[n]+'-stranger'+stranger[pid[n]][0])
        data_label.append(pid[n]+'-stranger'+stranger[pid[n]][1])

    nvalue = []
    edge_weights = []
    for n in range(N):
        nvalue.append([])
        edge_weights.append([])
        DGN.add_edge(pid[n], roommate[pid[n]][0])
        DGN.add_edge(pid[n], stranger[pid[n]][0])
        DGN.add_edge(pid[n], stranger[pid[n]][1])

        DG[n].add_edge(pid[n], roommate[pid[n]][0])
        DG[n].add_edge(pid[n], stranger[pid[n]][0])
        DG[n].add_edge(pid[n], stranger[pid[n]][1])

        nvalue[n].append(round(matrix[data_label[n*3+0]].to_list()[0]))
        nvalue[n].append(round(matrix[data_label[n*3+1]].to_list()[0]))
        nvalue[n].append(round(matrix[data_label[n*3+2]].to_list()[0]))

        edge_weights[n].append(nvalue[n][0]*5/std_coef) #weight of edge
        edge_weights[n].append(nvalue[n][1]*5/std_coef) #weight of edge
        edge_weights[n].append(nvalue[n][2]*5/std_coef) #weight of edge

        DG[n][pid[n]][roommate[pid[n]][0]]["weight"] = nvalue[n][0]
        DG[n][pid[n]][stranger[pid[n]][0]]["weight"] = nvalue[n][1]
        DG[n][pid[n]][stranger[pid[n]][1]]["weight"] = nvalue[n][2]

        #DG[n][pid[n]][roommate[pid[n]][0]]["fontcolor"] = nvalue[n][0]
        #DG[n][pid[n]][stranger[pid[n]][0]]["fontcolor"] = nvalue[n][1]
        #DG[n][pid[n]][stranger[pid[n]][1]]["fontcolor"] = nvalue[n][2]

    dpos = {}
    if N == 4:
        dpos[pid[0]] = (0,1)
        dpos[pid[1]] = (0,0)
        dpos[pid[2]] = (1,1)
        dpos[pid[3]] = (1,0)

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot()
    plt.title(title, fontsize=16)

    for n in range(N):
        edge_labels = {(i, j): w['weight'] for i, j, w in DG[n].edges(data=True)}
        nx.draw_networkx_edges(DG[n], dpos, edge_color=ncolor[n], width=edge_weights[n], arrowsize=25, arrows=True, alpha=0.9,connectionstyle="arc3, rad=0.2")
        nx.draw_networkx_edge_labels(DG[n], dpos, edge_labels=edge_labels, font_size=12, label_pos=0.3,clip_on=True, font_color=ncolor[n], font_family='Arial')
        nx.draw_networkx_labels(DG[n], dpos, font_size=12, font_color='w', )

    nx.draw_networkx_nodes(DGN, dpos, node_size=500, node_color=ncolor)
    nx.draw(DGN, dpos, ax, with_labels=False, font_size=12, node_size=500, node_color=ncolor, font_color='w', alpha=0)
    #plt.show()
    plt.savefig(setting['unit']+'_Day'+setting['phase']+'_'+title+'_network.jpg', bbox_inches="tight", dpi=180)  # 画像保存



def direct_graphnet1(pid, cb, pm,  matrix, w_coef, title, setting):
    N = len(pid)
    C = len(cb)
    P = len(pm)

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

    #Directed Graph object
    DG = nx.DiGraph()
    ncolor = ['blue', 'red', 'gray', 'orange']
    ecolor = ['blue', 'blue', 'blue', 'red', 'red', 'red', 'gray', 'gray', 'gray','orange','orange', 'orange']
    efcolor = {'blue', 'blue', 'blue', 'red', 'red', 'red', 'gray', 'gray', 'gray','orange','orange', 'orange'}

    data_label = []
    for n in range(N):
        DG.add_node(pid[n])
        data_label.append(pid[n]+'-roommate'+roommate[pid[n]][0])
        data_label.append(pid[n]+'-stranger'+stranger[pid[n]][0])
        data_label.append(pid[n]+'-stranger'+stranger[pid[n]][1])

    nvalue = []
    edge_weights = []
    for p in range(P):
        DG.add_edge(pid[pm[p][0]], pid[pm[p][1]])
        nvalue.append(round(matrix[data_label[p]].to_list()[0]))

        edge_weights.append(nvalue[p]/w_coef) #weight of edge

        DG[pid[pm[p][0]]][pid[pm[p][1]]]["weight"] = nvalue[p]
        DG[pid[pm[p][0]]][pid[pm[p][1]]]["fontcolor"] = ecolor[p]

    dpos = {}
    if N == 4:
        dpos[pid[0]] = (0,1)
        dpos[pid[1]] = (0,0)
        dpos[pid[2]] = (1,1)
        dpos[pid[3]] = (1,0)

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot()
    plt.title(title, fontsize=16)

    edge_labels = {(i, j): w['weight'] for i, j, w in DG.edges(data=True)}
    nx.draw_networkx_edge_labels(DG, dpos, edge_labels=edge_labels, font_size=12, label_pos=0.3,clip_on=False)
    nx.draw_networkx_labels(DG, dpos, font_size=12, font_color='w')
    nx.draw_networkx_nodes(DG, dpos, node_size=500, node_color=['blue', 'red','gray', 'orange'])
    nx.draw(DG, dpos, ax, with_labels=True, font_size=12, node_size=500, node_color=ncolor, edge_color=ecolor, font_color='w', width=edge_weights, alpha=.9, arrowsize=30, arrows=True, connectionstyle="arc3, rad=0.2", clip_on=True)
    #plt.show()
    plt.savefig(setting['unit']+'_Day'+setting['phase']+'_'+title+'_network.jpg', bbox_inches="tight", dpi=180)  # 画像保存
