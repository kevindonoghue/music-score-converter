import numpy as np
from data.generate_score import generate_score
from data.generate_bboxes import get_bboxes
from data.score_augmentations import random_score_augmentation
from skimage import io, transform
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import subprocess
import os
import pathlib
from PIL import Image, ImageDraw
import time





tight_tp_choices = ('wholes', 'halves_wholes', 'quarters', 'quarters_eighths')
wide_tp_choices = ('eighths', 'quarters_eighths', 'sixteenths', 'halves_quarters_eighths_sixteenths', 'quarters_eighths_sixteenths_dots', 'sixteenths')

cp_choices = ('none', 'tight', 'dense', 'complex')

rest_prob = np.random.choice([0, 0.2, 0.4, 0.6, 0.8], p=[0.5, 0.15, 0.15, 0.15, 0.05])


def generate_score_sample(sample_name, sample_size, num_measures,
                          treble_tp_key_choices, bass_tp_key_choices, treble_cp_key_choices, bass_cp_key_choices):
    t = time.time()
    pathlib.Path(f'{sample_name}/train/images/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{sample_name}/train/labels/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{sample_name}/val/images/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{sample_name}/val/labels/').mkdir(parents=True, exist_ok=True)


    for j in range(sample_size):
        if j % 5 == 0:
            phase = 'val'
        else:
            phase = 'train'
            
        rest_prob = np.random.choice([0, 0.2, 0.4, 0.6, 0.8], p=[0.5, 0.15, 0.15, 0.15, 0.05])
        measure_length = np.random.choice([8, 12, 16])
        key_number = 0
        
        with open(f'{sample_name}/temp.musicxml', 'w+') as f:
            f.write(str(generate_score(num_measures, measure_length, key_number, rest_prob, treble_tp_key_choices=treble_tp_key_choices, bass_tp_key_choices=bass_tp_key_choices, treble_cp_key_choices=treble_cp_key_choices, bass_cp_key_choices=bass_cp_key_choices)))
        subprocess.check_call(['mscore', '-S', str(pathlib.Path.home()) + '/Documents/MuseScore2/Styles/custom_style.mss', f'{sample_name}/temp.musicxml', '-o', f'{sample_name}/temp.mscx'])
        subprocess.check_call(['mscore', f'{sample_name}/temp.mscx', '-o', f'{sample_name}/temp.png'])
        subprocess.check_call(['mscore', f'{sample_name}/temp.mscx', '-o', f'{sample_name}/temp.svg'])

        i = 1
        while os.path.exists(f'{sample_name}/temp-{i}.png') and os.path.exists(f'{sample_name}/temp-{i}.svg'):
            filename = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=16))
            bbox_arr = get_bboxes(f'{sample_name}/temp-{i}.svg')
            heights = bbox_arr[:, 1, 1] - bbox_arr[:, 0, 1]
            if heights.max() > 0.25:
                i += 1
                continue
            page = io.imread(f'{sample_name}/temp-{i}.png')
            page = page[:, :, 3]/255
            page = 1 - page
            page, bbox_arr = random_score_augmentation(page, bbox_arr, 416, 416)
            bbox_arr = np.clip(bbox_arr, 0, 1)
            page = gray2rgb(page)
            page = (page*255).astype(np.uint8)
            io.imsave(f'{sample_name}/{phase}/images/{filename}.png', page)
            with open(f'{sample_name}/{phase}/labels/{filename}.txt', 'a+') as f:
                for k in range(bbox_arr.shape[0]):
                    x1 = bbox_arr[k, 0, 0]
                    y1 = bbox_arr[k, 0, 1]
                    x2 = bbox_arr[k, 1, 0]
                    y2 = bbox_arr[k, 1, 1]
                    center_x = (x2 + x1)/2
                    center_y = (y2 + y1)/2
                    width = (x2 - x1)
                    height = (y2 - y1)
                    f.write(f'0 {center_x} {center_y} {width} {height}\n')
                    
            image = Image.open(f'{sample_name}/{phase}/images/{filename}.png')
            draw = ImageDraw.Draw(image)
            for k in range(bbox_arr.shape[0]):
                x1 = bbox_arr[k, 0, 0]
                y1 = bbox_arr[k, 0, 1]
                x2 = bbox_arr[k, 1, 0]
                y2 = bbox_arr[k, 1, 1]
                bbox = ((x1*416, y1*416), (x2*416, y2*416))
                draw.rectangle(bbox, outline='red', width=5)
            image_np = np.array(image)
            plt.imshow(image_np, cmap='bone')
            plt.show()
            
            os.remove(f'{sample_name}/temp-{i}.png')
            os.remove(f'{sample_name}/temp-{i}.svg')    
            i += 1
        
        os.remove(f'{sample_name}/temp.musicxml')
        os.remove(f'{sample_name}/temp.mscx')
        print('time elapsed: ', time.time() - t)
            
        
        
    