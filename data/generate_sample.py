import numpy as np
import json
import os
import subprocess
import pathlib
from .crop import crop
from .generate_measure import generate_measure
from .xml_to_pc import xml_to_pc
from .augmentations import random_augmentation
import time



def generate_sample(sample_size, multiplicity, sample_name, height, width, treble_tp_key, bass_tp_key, treble_cp_key, bass_cp_key, rest_prob, measure_length_choices=(8, 12, 16), key_number_choices=tuple(range(-7, 8))):
    t = time.time()
    if not os.path.exists(sample_name + '/'):
        os.mkdir(sample_name + '/')
        
    pc_data = []
    images = []
    measure_lengths = []
    key_numbers = []
    
    for i in range(sample_size):
        print(f'{i}/{sample_size}    time: {time.time() - t} seconds')
        measure_length = np.random.choice(measure_length_choices)
        key_number = np.random.choice(key_number_choices)
        soup = generate_measure(measure_length, key_number, rest_prob, treble_tp_key, bass_tp_key, treble_cp_key, bass_cp_key)
        with open('temp.musicxml', 'w+') as f:
            f.write(str(soup))
        subprocess.call(['mscore', '-S', str(pathlib.Path.home()) + '/Documents/MuseScore2/Styles/custom_style.mss', 'temp.musicxml', '-o', 'temp.mscx'])
        subprocess.call(['mscore', 'temp.mscx', '-o', 'temp.png'])
        subprocess.call(['mscore', 'temp.mscx', '-o', 'temp.svg'])
        png_path = 'temp-1.png'
        svg_path = 'temp-1.svg'
        for _ in range(multiplicity):
            image = crop(png_path, svg_path)
            image = random_augmentation(image, height, width)
            image = (image*255).astype(np.uint8)
            images.append(image)
            measure_lengths.append(measure_length)
            key_numbers.append(key_number)


        # this line has to go down here since xml_to_pc changes soup in place
        pc = ['<START>'] + xml_to_pc(soup) + ['<END>']
        for _ in range(multiplicity):
            pc_data.append(pc)
        
        os.remove('temp.musicxml')
        os.remove('temp.mscx')
        os.remove('temp-1.png')
        os.remove('temp-1.svg')
        
    with open(sample_name + '/' + 'pc_data.json', 'w+') as f:
        json.dump(pc_data, f)
    measure_lengths = np.array(measure_lengths)
    key_numbers = np.array(key_numbers)
    images = np.array(images)
    np.save(sample_name + '/' + 'measure_lengths.npy', measure_lengths)
    np.save(sample_name + '/' + 'key_numbers.npy', key_numbers)
    np.save(sample_name + '/' + 'images.npy', images)
    
    info = {'sample_name': sample_name,
            'sample_size': sample_size,
            'multiplicity': multiplicity,
            'treble_tp_key': treble_tp_key,
            'bass_tp_key': bass_tp_key,
            'treble_cp_key': treble_cp_key,
            'bass_cp_key': bass_cp_key,
            'rest_prob': rest_prob,
            'measure_length_choices': list(measure_length_choices),
            'key_number_choices': list(key_number_choices)}
    with open(sample_name + '/' + 'info.json', 'w+') as f:
        json.dump(info, f)
        
    print(f'time elapsed: {time.time() - t} seconds')
    

