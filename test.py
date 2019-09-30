import numpy as np
import os
import json
import matplotlib.pyplot as plt

sample_path = 'quarters_eight_halves_chords_rests_c'

os.chdir(sample_path)
images = np.load('images.npy')
measure_lengths = np.load('measure_lengths.npy')
with open('pc_data.json') as f:
    pc_data = json.load(f)
    
print(images.shape)
n = np.random.randint(images.shape[0])
plt.imshow(images[n], cmap='bone')
plt.show()
print(pc_data[n])
print(measure_lengths[n])
print(np.min(list(map(len, pc_data))))
print(np.max(list(map(len, pc_data))))