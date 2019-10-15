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





# keys for randomized chord and rhythm choices (see generate_rhythm and generate_chords)
tight_tp_choices = ('wholes', 'halves_wholes', 'quarters', 'quarters_eighths')
wide_tp_choices = ('eighths', 'quarters_eighths', 'sixteenths', 'halves_quarters_eighths_sixteenths', 'quarters_eighths_sixteenths_dots', 'sixteenths')
cp_choices = ('none', 'tight', 'dense', 'complex')

# randomly pick a note/rest probability
rest_prob = np.random.choice([0, 0.2, 0.4, 0.6, 0.8], p=[0.5, 0.15, 0.15, 0.15, 0.05])


def generate_score_sample(sample_name, sample_size, num_measures,
                          treble_tp_key_choices, bass_tp_key_choices, treble_cp_key_choices, bass_cp_key_choices):
    """
    In the scores directory, creates a new directory called sample_name.
    Create subdirectories train and val each of which contains subdirectories images and labels.
    In each image folder, creates random images of pages of musical scores, resized down to shape (416, 416)
    In each labels folder, creates a text file describing the coordinates of bounding boxes around the measures in those pages.
    The names of the image and text files for the same page are the same.
    """
    t = time.time()
    pathlib.Path(f'scores/{sample_name}/train/images/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'scores/{sample_name}/train/labels/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'scores/{sample_name}/val/images/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'scores/{sample_name}/val/labels/').mkdir(parents=True, exist_ok=True)


    # train/val split of 0.8/0.2
    for j in range(sample_size):
        if j % 5 == 0:
            phase = 'val'
        else:
            phase = 'train'
            
        # randomly pick a note/rest probability
        rest_prob = np.random.choice([0, 0.2, 0.4, 0.6, 0.8], p=[0.5, 0.15, 0.15, 0.15, 0.05])
        measure_length = np.random.choice([8, 12, 16])
        key_number = 0
        
        # call generate_score to produce an xml file describing pages of music
        with open(f'scores/{sample_name}/temp.musicxml', 'w+') as f:
            f.write(str(generate_score(num_measures, measure_length, key_number, rest_prob, treble_tp_key_choices=treble_tp_key_choices, bass_tp_key_choices=bass_tp_key_choices, treble_cp_key_choices=treble_cp_key_choices, bass_cp_key_choices=bass_cp_key_choices)))
        
        # call musescore to convert the xml to a png and svg files
        subprocess.check_call(['mscore', '-S', str(pathlib.Path.home()) + '/Documents/MuseScore2/Styles/custom_style.mss', f'scores/{sample_name}/temp.musicxml', '-o', f'scores/{sample_name}/temp.mscx'])
        subprocess.check_call(['mscore', f'scores/{sample_name}/temp.mscx', '-o', f'scores/{sample_name}/temp.png'])
        subprocess.check_call(['mscore', f'scores/{sample_name}/temp.mscx', '-o', f'scores/{sample_name}/temp.svg'])

        # loop over all the pages created from the xml file
        i = 1
        while os.path.exists(f'scores/{sample_name}/temp-{i}.png') and os.path.exists(f'scores/{sample_name}/temp-{i}.svg'):
            # pick a random filename
            filename = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=16))
            
            # use the svg file to draw bounding boxes around the measures
            bbox_arr = get_bboxes(f'scores/{sample_name}/temp-{i}.svg')
            
            # ignore pages that are too sparse
            heights = bbox_arr[:, 1, 1] - bbox_arr[:, 0, 1]
            if heights.max() > 0.25:
                i += 1
                continue
            
            # augment and resize the .png file of the page
            page = io.imread(f'scores/{sample_name}/temp-{i}.png')
            page = page[:, :, 3]/255
            page = 1 - page
            if page.shape[0] == 0:
                i += 1
                continue
            page, bbox_arr = random_score_augmentation(page, bbox_arr, 416, 416)
            bbox_arr = np.clip(bbox_arr, 0, 1)
            page = gray2rgb(page)
            page = (page*255).astype(np.uint8)
            
            # save the bbox data and the image of the page
            io.imsave(f'scores/{sample_name}/{phase}/images/{filename}.png', page)
            with open(f'scores/{sample_name}/{phase}/labels/{filename}.txt', 'a+') as f:
                for k in range(bbox_arr.shape[0]):
                    # the bbox data is stored in the text file following the instructions at https://github.com/ultralytics/yolov3
                    x1 = bbox_arr[k, 0, 0]
                    y1 = bbox_arr[k, 0, 1]
                    x2 = bbox_arr[k, 1, 0]
                    y2 = bbox_arr[k, 1, 1]
                    center_x = (x2 + x1)/2
                    center_y = (y2 + y1)/2
                    width = (x2 - x1)
                    height = (y2 - y1)
                    f.write(f'0 {center_x} {center_y} {width} {height}\n')
                
            # # uncomment to visually debug bounding boxes    
            # image = Image.open(f'scores/{sample_name}/{phase}/images/{filename}.png')
            # draw = ImageDraw.Draw(image)
            # for k in range(bbox_arr.shape[0]):
            #     x1 = bbox_arr[k, 0, 0]
            #     y1 = bbox_arr[k, 0, 1]
            #     x2 = bbox_arr[k, 1, 0]
            #     y2 = bbox_arr[k, 1, 1]
            #     bbox = ((x1*416, y1*416), (x2*416, y2*416))
            #     draw.rectangle(bbox, outline='red', width=5)
            # image_np = np.array(image)
            # plt.imshow(image_np, cmap='bone')
            # plt.show()
            
            # remove unneeded files
            os.remove(f'scores/{sample_name}/temp-{i}.png')
            os.remove(f'scores/{sample_name}/temp-{i}.svg')    
            i += 1
        
        # remove unneeded files
        os.remove(f'scores/{sample_name}/temp.musicxml')
        os.remove(f'scores/{sample_name}/temp.mscx')
        print('time elapsed: ', time.time() - t)
            
        
        
    