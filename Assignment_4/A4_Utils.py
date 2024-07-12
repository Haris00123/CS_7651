import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from bettermdptools.algorithms.rl import RL
from bettermdptools.utils.test_env import TestEnv
import time
import csv
import pygame
from scipy import stats

def mode_policy_dict(policies,gammas):
    '''Function to compute optimal policy for each gamma value using mode'''
    mode_policies=np.zeros(shape=(gammas.shape[0],len(policies[list(policies.keys())[0]])))
    for i,g in enumerate(gammas):
        master_key='{:.2f}_'.format(g)
        big_array=None
        short_listed_keys=[]
        for k in policies.keys():
            if master_key in k:
                short_listed_keys.append(k)
            else:
                pass
        for ii,s_k in enumerate(short_listed_keys):
            if ii==0:
                big_array=np.array(list(policies[s_k].values()))
            else:
                big_array=np.c_[big_array,np.array(list(policies[s_k].values()))]
            
        #Getting Mode
        mode_policies[i]=stats.mode(big_array,axis=1)[0].ravel()
    return mode_policies

def visualize_policy(policies,env,gammas,title_header='None',row=3,cols=5):
    #env=[v.decode('ascii') for v in env]
    env=env.ravel()
    fig, axes = plt.subplots(row,cols)
    fig.set_size_inches(10*cols,5*row)
    plt.suptitle(f'{title_header} Policies Plotted',fontsize=18)
    arrow_colors = {
    0: 'blue',
    1: 'black',
    2: 'red',
    3: 'green'
    }

    grid_colors={
        b'H': 'red',
        b'G': 'green',
        b'F': 'grey',
        b'S': 'grey'
    }

    for i,ax in enumerate(axes.flatten()):
        policy=policies[i]
        ax.set_title('Gamma: {:.2f}'.format(gammas[i]))
        ax.set_xticks(np.arange(policy.shape[0]**(0.5)))
        ax.set_yticks(-1*np.arange(policy.shape[0]**(0.5)))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

        for ii in range(len(policy)):
            if policy[ii]==2:
                arrow = '→'
            elif policy[ii]==0:
                arrow = '←'
            elif policy[ii]==3:
                arrow = '↑'
            elif policy[ii]==1:
                arrow = '↓'
            else:
                continue
            x=int(ii%policy.shape[0]**(0.5))
            y=-int(ii/policy.shape[0]**(0.5))
            rect = plt.Rectangle((x, y), 1, 1, color=grid_colors[env[ii]], alpha=0.1)
            ax.add_patch(rect)
            ax.text(x,y,arrow, ha='center', va='center', color=arrow_colors[policy[ii]])
    plt.tight_layout()
    plt.show()

def visualize_heatmap_states(values,gammas,title_header='None',row=3,cols=5):
    #env=[v.decode('ascii') for v in env]
    fig, axes = plt.subplots(row,cols)
    fig.set_size_inches(10*cols,5*row)
    plt.suptitle(f'{title_header} Policies Plotted',fontsize=18)

    for i,ax in enumerate(axes.flatten()):
        value=values[i,:]
        ax.set_title('Gamma: {:.2f}'.format(gammas[i]))
        ax.set_xticks(np.arange(values.shape[0]**(0.5)))
        ax.set_yticks(-1*np.arange(values.shape[0]**(0.5)))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        heatmap=ax.imshow(value.reshape(int(len(value)**(0.5)),int(len(value)**(0.5))), cmap='jet', interpolation='lanczos')
        plt.colorbar(heatmap, ax=ax)
    plt.tight_layout()
    plt.show()