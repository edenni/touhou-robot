import random
import utils
import PIL
import cv2
import numpy as np
from PIL import Image
import pyocr
import os
import time
from typing import Dict, Tuple, List, Iterable
from skimage.metrics import structural_similarity
from utils import pil2cv
import matplotlib.pyplot as plt

# health memory: 004b5a40
# score: 004B5A0C
# jugde death code:  00447dc4
# dec health : 0044921a


def cv2show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def is_death(img, region, mask, threshold):
    '''deal with death screen
    img: np.ndarray HWC
    region: (y1, y2, x1, x2)
    mask: np.ndarray HW.   white text with black background
    threshold: int  sum of pixel value / 255 after masking
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    y1, y2, x1, x2 = region
    img = img[y1:y2, x1:x2]
    _, text_mask = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    if np.sum(text_mask/255) < threshold:
        return False
    text = cv2.bitwise_and(img, img, mask=text_mask)
    # cv2.imwrite("ff.bmp", text)
    # plt.imshow(text, cmap='gray')
    # plt.show()

    score, diff = structural_similarity(text, mask, full=True)
    return score > 0.95

def death_scene(img: np.ndarray, 
        regions_masks: Dict[str, Tuple[Tuple, np.ndarray]], 
        threshold=50):
    '''return None if not dead else name of death scene 
    '''
    for name, (region, mask) in regions_masks.items():
        if is_death(img, region, mask, threshold):
            return name
    return None

def get_regions_masks_from_names(roi : Tuple[str], 
        path = './data') -> Dict[str, Tuple[Tuple, np.ndarray]]:
   
    d = {}
    for name in roi:
        region = regions[name]
        # mask is None if not find img
        mask = cv2.imread(f'{path}/{name}.bmp', cv2.IMREAD_GRAYSCALE)
        d[name] = (region, mask)

    return d

regions = {
    'manshinsoui': (400, 450, 75, 245),
    'credit': (880, 960, 360, 500),
    'score': (130, 160, 290, 590),
    'main': (920, 960, 440, 840),
}
    
def scene_solver(death_scene):
    if not death_scene:
        return None
    return Solver(death_scene)
