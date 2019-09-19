import numpy as np
from xml_to_pc import xml_to_pc
from generate_measure import generate_measure
import json
import os

font_names = ['emmentaler', 'bravura', 'gonville', 'musejazz']

for font_name in font_names:
    extra_data = []
    if not os.path.exists(f'generated_data_{font_name}'):
        os.mkdir(f'generated_data_{font_name}')
    for i in range(10):
        chord_probs = np.array([5, 1, 5, 5, 5, 5, 1, 100, 1, 5, 5, 5, 5, 1, 5])
        chord_probs = chord_probs / chord_probs.sum()
        measure_length = int(4*np.random.choice([3, 4]))
        rest_prob = 0.2
        key_number = int(np.random.randint(-7, 8))
        soup = generate_measure(measure_length, key_number, rest_prob, chord_probs)
        with open(f'generated_data_{font_name}/{i}_{font_name}.musicxml', 'w+') as f:
            f.write(str(soup))
        # xml_to_pc modifies soup in place
        # unfortunately, BeautifulSoup doesn't seem to have a cloning method
        pc = ['<START>'] + xml_to_pc(soup) + ['<END>']
        extra_data.append({'measure_length': measure_length, 'key_number': key_number, 'pc': pc})
        
    with open(f'generated_data_{font_name}/extra_data_{font_name}.json', 'w+') as f:
        json.dump(extra_data, f)
