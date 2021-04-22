"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
import random
from utils import read_db, observe

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 8  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    def __init__(self, is_train = False,test_folder='',is_monitor = False, reward_mag=1.0):
        super(Maze, self).__init__()
        self.action_space = ['0','1','2','3','4','5','6','7']
        self.n_actions = len(self.action_space)
        self.max_passes = 3
        self.title('State space')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self.origin = np.array([0, 0, 0])
        self.cur_state = np.array([0, 0, 0])
        self.oval_state = np.array([0, 0, 0])
        # read training dataset
        self.is_train = is_train
        self.is_monitor = is_monitor
        self.test_folder = test_folder
        self.reward_mag = reward_mag
        self.db = read_db(is_train = self.is_train, test_folder=self.test_folder)
        self.ob = observe(self.db.sv,self.db.cp,self.db.db)
        self._build_maze() 
        # test optimal rate
        self.hit_top_1 = 0
        self.hit_top_3 = 0
        self.hit_top_5 = 0
    
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # set origin and oval (goal)
        self.cur_state = self.origin.copy()  
        self.oval_state = self.origin.copy()
        #print('OPT:',self.oval_state)
        # print('oval_State', self.oval_state)
        # draw environment
        origin_center = np.array([np.sum(self.origin!=0)+int(.5*UNIT),np.max(self.origin)+int(.5*UNIT)])
        #print(origin_center)
        cur_center = np.array([np.sum(self.cur_state!=0)+int(.5*UNIT),np.max(self.cur_state)+int(.5*UNIT)])
        oval_center = origin_center + np.array([np.sum(self.oval_state!=0),np.max(self.oval_state)])*UNIT
        #print(oval_center)
        #print(np.sum(self.oval_state!=0))
        # create goal
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create agent
        self.rect = self.canvas.create_rectangle(
            cur_center[0] - 15, cur_center[1] - 15,
            cur_center[0] + 15, cur_center[1] + 15,
            fill='red')
        
        for i in range(int(.5*UNIT)+UNIT, int(.5*UNIT)+(MAZE_H)*UNIT, UNIT):
            self.cross = self.canvas.create_line(20-15, i-15, 20+15, i+15)
            self.cross = self.canvas.create_line(20+15, i-15, 20-15, i+15)

        # pack all
        self.canvas.pack()

    def reset(self):
        # print('----reset env')
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect)
        self.canvas.delete(self.oval)
        self.cur_state = self.origin.copy()
        self.hit_top_1 = 0
        self.hit_top_3 = 0
        self.hit_top_5 = 0
        #print('oval_state', self.oval_state)
        #print('cur_state', self.cur_state)
        # new observation
        if not self.is_train:
            self.ob.sequence_select()
            self.ob.print_training_data_infos()
        else:
            self.ob.random_settings()
        
        if self.is_monitor:
            self.ob.print_training_data_infos()
        self.oval_state = np.asarray(self.ob.r_data.iloc[[np.argmax(np.asarray(self.ob.r_data.CEL))]][['p1','p2','p3']])[0]
        #print('OPT:',self.oval_state)
        
        origin_center = np.array([np.sum(self.origin!=0)+.5*UNIT,np.max(self.origin)+.5*UNIT])
        cur_center = np.array([np.sum(self.cur_state!=0)+.5*UNIT,np.max(self.cur_state)+.5*UNIT])
        oval_center = origin_center + np.array([np.sum(self.oval_state!=0),np.max(self.oval_state)])*UNIT
        
        # create oval
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            cur_center[0] - 15, cur_center[1] - 15,
            cur_center[0] + 15, cur_center[1] + 15,
            fill='red')
        
        # return observation
        return [self.cur_state, self.ob.I_src, self.ob.D_src, self.ob.P_src, self.ob.O_src, self.ob.P_tar, self.ob.O_tar]

    def step(self, action):
        # get current state
        cur_passes = np.sum(self.cur_state!=0)
        cur_views = np.max(self.cur_state)
    
        # update next state
        next_state = self.cur_state.copy()
        next_state[cur_passes] = cur_views+action
        done = False
        reward = 0
        
        # print("{0} -> {1}".format(self.cur_state, next_state))

        # End check
        if action == 0: # choose action 0
            if (self.cur_state == self.origin).all(): # penalizing stopping at original state 
                reward = -10
            next_state[cur_passes] = 0
            done = True
            # print('done action == 0')
        elif np.max(next_state) > (self.n_actions - 1) : # source exhaust
            done = True
            reward = -1
            # print('done p.max(next_state) >=7')
        elif np.sum(next_state!=0) == (self.max_passes): # reach the passes 3
            done = True
            reward = np.asarray(self.ob.r_data.loc[(self.ob.r_data.p1 == next_state[0]) & (self.ob.r_data.p2 == next_state[1]) & (self.ob.r_data.p3 == next_state[2])].CEL) - np.asarray(self.ob.r_data.loc[(self.ob.r_data.p1 == self.cur_state[0]) & (self.ob.r_data.p2 == self.cur_state[1]) & (self.ob.r_data.p3 == self.cur_state[2])].CEL)
            # print('done np.sum(next_state!=0)>=3')
        else:
            if (self.cur_state == self.origin).all():
                reward = np.asarray(self.ob.r_data.loc[(self.ob.r_data.p1 == next_state[0]) & (self.ob.r_data.p2 == next_state[1]) & (self.ob.r_data.p3 == next_state[2])].CEL)
            else:
                reward = np.asarray(self.ob.r_data.loc[(self.ob.r_data.p1 == next_state[0]) & (self.ob.r_data.p2 == next_state[1]) & (self.ob.r_data.p3 == next_state[2])].CEL) - np.asarray(self.ob.r_data.loc[(self.ob.r_data.p1 == self.cur_state[0]) & (self.ob.r_data.p2 == self.cur_state[1]) & (self.ob.r_data.p3 == self.cur_state[2])].CEL)
        
        # reward
        reward = reward*self.reward_mag
        
        top_5_setting = sorted(zip(np.asarray(self.ob.r_data.CEL),np.asarray(self.ob.r_data.p1),np.asarray(self.ob.r_data.p2),np.asarray(self.ob.r_data.p3)), reverse=True)[:5]
        hit_table = (next_state == np.asarray(top_5_setting)[:5,1:]).all(axis=1)
        
        if self.is_monitor:
            print ("s: {0}, a:{1}, reward:{2}, s_:{3}, hit_table{4}".format(self.cur_state,action,reward,next_state, hit_table))
            
        if hit_table[:1].any():
            self.hit_top_1 = 1
        if hit_table[:3].any():
            self.hit_top_3 = 1
        if hit_table[:5].any():
            self.hit_top_5 = 1
        #
        # draw animation
        origin_center = np.array([np.sum(self.origin!=0)+.5*UNIT,np.max(self.origin)+.5*UNIT])
        next_center = np.array([np.sum(next_state!=0)-cur_passes,np.max(next_state)-cur_views])*UNIT
                
        self.canvas.move(self.rect, next_center[0], next_center[1])  # move agent 
        s_ = next_state
        self.cur_state = next_state
        return s_, reward, done

    def render(self):
        time.sleep(0.001)
        self.update()
