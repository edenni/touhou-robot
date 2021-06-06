import time
import ctypes
import win32process
import win32con
import win32api
import ctypes
import win32gui
from typing import Tuple

import cv2
import numpy as np
import gym
import logging

import config
from helper import (get_regions_masks_from_names, 
                   death_scene, is_death)
from utils import (get_screen, presskey, pil2cv)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler('log.txt')
handler.setFormatter(logging.Formatter("[%(levelname)8s]-%(asctime)s  %(message)s"))
logger.addHandler(handler)

class THEnv(gym.Env):

    def __init__(self, roi : Tuple[str]):
        super(THEnv, self).__init__()
        self.total_death = 0
        self.masks = get_regions_masks_from_names(roi)

        PROCESS_ALL_ACCESS = (0x000F0000 | 0x00100000 | 0xFFF)
        window = win32gui.FindWindow(0, "東方鬼形獣　～ Wily Beast and Weakest Creature. ver 1.00b")
        hid, pid = win32process.GetWindowThreadProcessId(window)
        self.phand = win32api.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        self.num_death = ctypes.c_long() 
        self.score = ctypes.c_int32()
        self.mydll = ctypes.windll.LoadLibrary("C:\\Windows\\System32\\kernel32.dll")

        self.death_actions = {
            'manshinsoui': ['z']+['delay']*3,
            'score': ['z', 'delay', 'z'],
            'main': ['z', 'delay']*4+['delay']*3,}
        attack = ['z']
        slow = ['lshift']
        move = [['up'], ['down'], ['left'], ['right'], ['up', 'right'], ['up', 'left'],
                ['down', 'left'], ['down', 'right']]
        self.actions = [[], attack+slow,
            *[attack+m for m in move], 
            *[attack+slow+m for m in move]]

        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(low=0, high=1, 
                                shape=(84, 84), dtype=np.float)
        self.info = {'episode': 0}

    def step(self, action):
        is_dead = False
        menu = None

        if self._is_dead():
            self.total_death += 1
            time.sleep(3)
            menu = self.leave_menu()
            is_dead = True
     
        logger.info('state: '+ str(menu)+' zanki: ' + str(self.num_death.value) + ' total_death: ' + str(self.total_death) + ' action: ' + ' '.join(self.actions[action]))
     
        if not is_dead:
            presskey(self.actions[action])

        next_state = pil2cv(self.get_screen().crop((70, 60, 830, 930)))
        next_state = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
        next_state = cv2.resize(next_state, config.image_size)

        reward = 0.1
        score = self._inc_score()
        if score > 0:
            reward += score
        if is_dead:
            reward = -100

        return next_state / 255.0, reward, menu is not None, {}

    def _is_dead(self):
        pre_data = self.num_death.value
        self.mydll.ReadProcessMemory(int(self.phand), 0x004b5a40, ctypes.byref(
            self.num_death), 4, None)

        return self.num_death.value < pre_data

    def _inc_score(self):
        pre_data = self.score.value
        self.mydll.ReadProcessMemory(int(self.phand), 0x004B5A0C, ctypes.byref(
            self.score), 4, None)
        return self.score.value - pre_data

    @classmethod
    def get_screen(cls, x=80, y=30, w=1280, h=960):
        return get_screen(x, y, w, h)

    def reset(self):
        '''leave out the menu and start game
        '''
        self.leave_menu()
        self.info['episode'] += 1
        state = cv2.resize(pil2cv(self.get_screen().crop((70, 60, 830, 930))), config.image_size)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        return state / 255.0

    def leave_menu(self, img=None, delay=1):
        if img is None or not isinstance(img, np.ndarray):
            img = self.get_screen()
            img = pil2cv(img)

        scene = death_scene(img, self.masks)
        result = scene

        while scene:
            actions = self.death_actions[scene]
            for key in actions:
                if key == 'delay':
                    time.sleep(delay)
                    continue
                presskey(key, interval=0.25)

            img = self.get_screen()
            img = pil2cv(img)
            scene = death_scene(img, self.masks)
            if scene: result = scene
        
        for i in range(5):
            presskey(['z'])
        return result
