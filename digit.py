
import os
import cv2
import pyocr
from PIL import Image
import numpy as np


#https://www.slideshare.net/TakeshiHasegawa1/20151016ssmjpikalog

path_tesseract = "C:\\Program Files\\Tesseract-OCR"
if path_tesseract not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + path_tesseract
tool = pyocr.get_available_tools()[0]
builder = pyocr.builders.WordBoxBuilder(6)

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(
                new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(
                new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = Image.open('.\data\death\\0012.jpg')
# res = tool.image_to_string(img, 
#   lang='eng', builder=builder)
img = pil2cv(img)
img = img[120:180, 1000:1280]
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

ret, thresh = cv2.threshold(img_gray, 
    200, 255, cv2.THRESH_BINARY)
cv2show(thresh)

contours, _ = cv2.findContours(thresh, 
    cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)
# print(contour)
digits = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if max(w, h) / min(w, h) > 10 or min(w, h) < 10:
        continue
    cv2.rectangle(img, (x, y), 
        (x+w, y+h), (0, 0, 255), 2)
    digit = img_gray[y:y+h, x:x+w]
    resized_digit = cv2.resize(digit, (18, 18))
    padded_digit = np.pad(resized_digit, 
        ((5, 5), (5, 5)))
    digits.append(padded_digit)

processed_img = cv2.Canny(img, 50, 100)
cv2show(processed_img)

# for d in res:
    # cv2.rectangle(img, d.position[0], d.position[1], (0, 0, 255), 2)
cv2show(img)
