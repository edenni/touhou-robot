import ctypes
import win32gui
import time
from time import sleep

from PIL import ImageGrab
import cv2
import numpy as np

# https://docs.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-keybdinput
# hardware scan code: https://so-zou.jp/pc/keyboard/scan-code.htm


keys = {
    'esc':      0x01,
    'z':        0x2c,
    'x':        0x2d,
    'left':     0xcb,
    'up':       0xc8,
    'right':    0xcd,
    'down':     0xd0,
    'lshift':   0x2a,
    'enter':    0x1c,
}

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def _presskey(key):
    key = key.lower()
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, keys[key], 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def releasekey(key):
    key = key.lower()
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, keys[key], 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラーz
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def get_screen(x, y, w = 1280, h = 960):
    '''take screenshot
    '''
    img = ImageGrab.grab((x, y, x+w, y+h))
    return img
    
def presskey(keys, interval=0.02):
    for k in keys:
        _presskey(k)
    sleep(interval)
    for k in keys[::-1]:
        releasekey(k)

if __name__ == "__main__":
    # presskey('z', 'x')
    # img = get_screen(80, 30)
    # from helper import death_scene, get_regions_masks_from_names, pil2cv
    # from config import roi
    # mask = get_regions_masks_from_names(roi)
    # img.show()
    # death_scene(pil2cv(img), mask)
    stime = time.time()
    count = 0
    while True:
        get_screen(80, 30)
        count += 1
        if time.time() - stime > 10:
            break
    print(f'fps: {count / 10}')