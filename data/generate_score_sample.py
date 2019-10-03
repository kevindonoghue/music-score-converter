import numpy as np
from data.generate_score import generate_score
from data.generate_bboxes import get_bboxes
from skimage import io, transform
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import subprocess
import os
import pathlib
from PIL import Image, ImageDraw





tight_tp_choices = ('wholes', 'halves_wholes', 'quarters', 'quarters_eighths')
wide_tp_choices = ('eighths', 'quarters_eighths', 'sixteenths', 'halves_quarters_eighths_sixteenths', 'quarters_eighths_sixteenths_dots', 'sixteenths')

cp_choices = ('none', 'tight', 'dense', 'complex')

measure_length = 16
key_number = 0
rest_prob = np.random.choice([0, 0.2, 0.4, 0.6, 0.8], p=[0.5, 0.15, 0.15, 0.15, 0.05])
treble_tp_key_choices = tight_tp_choices
bass_tp_key_choices = wide_tp_choices
treble_cp_key_choices = cp_choices
bass_cp_key_choices = cp_choices

sample_name = 'asdf'
sample_size = 10
pathlib.Path(f'{sample_name}/train/images/').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'{sample_name}/train/labels/').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'{sample_name}/test/images/').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'{sample_name}/test/labels/').mkdir(parents=True, exist_ok=True)

if not os.path.exists(f'{sample_name}/'):
    os.mkdir(f'{sample_name}/')
if not os.path.exists(f'{sample_name}/images/'):
    os.mkdir(f'{sample_name}/images/')
if not os.path.exists(f'{sample_name}/labels/'):
    os.mkdir(f'{sample_name}/labels/')


for j in range(sample_size):
    if j % 5 == 0:
        phase = 'val'
    else:
        phase = 'train'
    with open(f'{sample_name}/temp.musicxml', 'w+') as f:
        # f.write(str(generate_score(64, measure_length, key_number, rest_prob, treble_tp_key_choices=treble_tp_key_choices, bass_tp_key_choices=bass_tp_key_choices, treble_cp_key_choices=treble_cp_key_choices, bass_cp_key_choices=bass_cp_key_choices)))
        f.write(str(generate_score(64, measure_length, key_number, rest_prob, treble_tp_key_choices=('quarters',), bass_tp_key_choices=('quarters',), treble_cp_key_choices=('none',), bass_cp_key_choices=('none',))))
        

    subprocess.call(['mscore', '-S', str(pathlib.Path.home()) + '/Documents/MuseScore2/Styles/custom_style.mss', f'{sample_name}/temp.musicxml', '-o', f'{sample_name}/temp.mscx'])
    subprocess.call(['mscore', f'{sample_name}/temp.mscx', '-o', f'{sample_name}/temp.png'])
    subprocess.call(['mscore', f'{sample_name}/temp.mscx', '-o', f'{sample_name}/temp.svg'])

    i = 1
    while os.path.exists(f'{sample_name}/temp-{i}.png') and os.path.exists(f'{sample_name}/temp-{i}.svg'):
        filename = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=16))
        bboxes = get_bboxes(f'{sample_name}/temp-{i}.svg')
        page = io.imread(f'{sample_name}/temp-{i}.png')
        page = page[:, :, 3]
        page = 255 - page
        page = transform.resize(page, (416, 416))
        page = gray2rgb(page)
        io.imsave(f'{sample_name}/{phase}/images/{filename}.png', page)
        with open(f'{sample_name}/{phase}/labels/{filename}.txt', 'a+') as f:
        for row in bboxes:
            for (x1, y1), (x2, y2) in row:
                center_x = (x2 + x1)/2
                center_y = (y2 + y1)/2
                width = (x2 - x1)/2
                height = (y2 - y1)/2
                f.write(f'0 {center_x} {center_y} {width} {height}\n')
        os.remove(f'{sample_name}/temp-{i}.png')
        os.remove(f'{sample_name}/temp-{i}.svg')    
        i += 1

    
    
    