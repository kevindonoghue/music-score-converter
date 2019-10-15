import numpy as np
import os
import subprocess
import sys
from skimage import io
from skimage.color import gray2rgb
from yolov3_detect.detect import detect
from django.conf import settings

MEDIA_ROOT = settings.MEDIA_ROOT
BASE_DIR = settings.BASE_DIR
YOLO_DIR = os.path.join(BASE_DIR, 'yolov3_detect')

os.makedirs(os.path.join(MEDIA_ROOT, 'yolo_output'), exist_ok=True)
os.makedirs(os.path.join(MEDIA_ROOT, 'current_measures'), exist_ok=True)

class Opt:
    """
    I forked the yolo implementation from https://github.com/ultralytics/yolov3.
    In order to use their scripts, I need to create this helper class.
    It stores the information that would be passed by a command line parser.
    """
    def __init__(self, path):
        self.source = path
        self.cfg = os.path.join(YOLO_DIR, 'cfg', 'score_yolo.cfg')
        self.weights = os.path.join(YOLO_DIR, 'weights', 'last-2019-10-05.pt')
        self.output = os.path.join(MEDIA_ROOT, 'yolo_output')
        self.data = os.path.join(YOLO_DIR, 'data', 'score_yolo.data')
        self.img_size = 416
        self.conf_thres = 0.3
        self.nms_thres = 0.5
        self.fourcc = 'mp4v'
        self.half = False
        self.device = ''
        self.view_img = False
        

def handle_page(path, measure_length, key_number, dir):
    """
    path is the path to the image of the page uploaded by the user.
    This function detects the measures in that image, crops them out,
    and saves them in the media/current_measures directory.
    """
    filename = os.path.basename(path)[:-4]
    for file in os.scandir(os.path.join(MEDIA_ROOT, 'current_measures')):
        os.remove(file)
    opt = Opt(path)
    detect(opt)
    image = io.imread(os.path.join(MEDIA_ROOT, filename + '.png'))
    image_with_boxes = io.imread(os.path.join(MEDIA_ROOT, 'yolo_output', filename + '.png'))
    height, width = image.shape[0], image.shape[1]
    if len(image.shape) == 2:
        image = gray2rgb(image)
    with open(os.path.join(MEDIA_ROOT, 'yolo_output', filename + '.txt')) as f:
        bbox_string = f.read()
    lines = bbox_string.split('\n')[:-1]
    coordinates = [tuple(map(int, x.split()[:4])) for x in lines if float(x.split()[5])>0.8]
    coordinates = sorted(coordinates, key=lambda tup: tup[1])
    rows = []
    rows.append([coordinates[0]])
    for x in coordinates[1:]:
        last = rows[-1][-1]
        height = last[3] - last[1]
        if x[1] > last[1] + height/2:
            rows.append([x])
        else:
            rows[-1].append(x)
    sorted_coordinates = []
    for row in rows:
        row.sort(key=lambda t: t[0])
        sorted_coordinates.extend(row)
    print(sorted_coordinates)
    for i, (x1, y1, x2, y2) in enumerate(sorted_coordinates):
        print('cropping image ', i+1)
        subimage = image[y1:y2, x1:x2]
        mins = subimage.min(axis=2).min(axis=1)
        top_limit = int(np.argmax(mins < 250)/2)
        bottom_limit = len(mins) - int(np.argmax(np.flip(mins) < 250)/2)
        subimage = subimage[top_limit:bottom_limit]
        target_dir = os.path.join(MEDIA_ROOT, 'current_measures')
        io.imsave(os.path.join(target_dir, f'subimage{i}.png'), subimage)



# handle_page('minuet_larger2.png', 12, 0, None)