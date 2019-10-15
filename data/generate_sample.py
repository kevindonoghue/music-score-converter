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
    """
    Generates a sample of measures, to be fed into the neural net.
    - sample_size is the number of distinct images of measures
    - multiplicity is the number of times that measure should be augmented, so the actual samples size is sample_size*multiplicity
    - sample_name is the name of the folder in which the sample data is deposited
    - height and width are the heights and widths of the images of measures
    - for the keys, see the files generate_rhtyhm and generate_chords
    - rest_prob indicates the frequency of rests vs notes
    - the sampler randomly selects from the given key and measure number choices
    """
    
    t = time.time()
    if not os.path.exists(sample_name + '/'):
        os.mkdir(sample_name + '/')
        
    # the neural net needs images of measures, pseudocode for those images, and the key and time signatures of those measures
    pc_data = []
    images = []
    measure_lengths = []
    key_numbers = []
    
    for i in range(sample_size):
        print(f'{i}/{sample_size}    time: {time.time() - t} seconds')
        
        # pick random key and time signatures
        measure_length = np.random.choice(measure_length_choices)
        key_number = np.random.choice(key_number_choices)
        
        # generate the xml for a measure
        soup = generate_measure(measure_length, key_number, rest_prob, treble_tp_key, bass_tp_key, treble_cp_key, bass_cp_key)
        
        # write the xml to a file
        with open('temp.musicxml', 'w+') as f:
            f.write(str(soup))
            
        # call musescore to convert the xml to a png and and svg
        subprocess.call(['mscore', '-S', str(pathlib.Path.home()) + '/Documents/MuseScore2/Styles/custom_style.mss', 'temp.musicxml', '-o', 'temp.mscx'])
        subprocess.call(['mscore', 'temp.mscx', '-o', 'temp.png'])
        subprocess.call(['mscore', 'temp.mscx', '-o', 'temp.svg'])
        png_path = 'temp-1.png'
        svg_path = 'temp-1.svg'
        
        # create multiple copies of the measure, with the image distorted in different ways
        for _ in range(multiplicity):
            # use the png and svg to appropriately crop the image
            image = crop(png_path, svg_path)
            
            # distort the image
            image = random_augmentation(image, height, width)
            
            # make sure the image is encoded as an integer, to save space
            image = (image*255).astype(np.uint8)
            
            # add the image data to the lists
            images.append(image)
            measure_lengths.append(measure_length)
            key_numbers.append(key_number)

        # record the pseucode, instead of the xml
        # this line has to go down here since xml_to_pc changes soup in place
        pc = ['<START>'] + xml_to_pc(soup) + ['<END>']
        for _ in range(multiplicity):
            pc_data.append(pc)
        
        # remove the files used in constructing the sample item
        os.remove('temp.musicxml')
        os.remove('temp.mscx')
        os.remove('temp-1.png')
        os.remove('temp-1.svg')
        
    # after generating the sample, save it to a folder
    with open(sample_name + '/' + 'pc_data.json', 'w+') as f:
        json.dump(pc_data, f)
    measure_lengths = np.array(measure_lengths)
    key_numbers = np.array(key_numbers)
    images = np.array(images)
    np.save(sample_name + '/' + 'measure_lengths.npy', measure_lengths)
    np.save(sample_name + '/' + 'key_numbers.npy', key_numbers)
    np.save(sample_name + '/' + 'images.npy', images)
    
    time_elapsed = time.time() - t
    
    # save info about the sample to a text file
    info = {'sample_name': sample_name,
            'sample_size': sample_size,
            'multiplicity': multiplicity,
            'treble_tp_key': treble_tp_key,
            'bass_tp_key': bass_tp_key,
            'treble_cp_key': treble_cp_key,
            'bass_cp_key': bass_cp_key,
            'rest_prob': rest_prob,
            'measure_length_choices': list(measure_length_choices),
            'key_number_choices': list(key_number_choices),
            'time_elapsed': time_elapsed}
    with open(sample_name + '/' + 'info.json', 'w+') as f:
        json.dump(info, f)

