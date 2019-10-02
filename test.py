import numpy as np
from data.generate_score import generate_score


tight_tp_choices = ('wholes', 'halves_wholes', 'quarters', 'quarters_eighths')
wide_tp_choices = ('eighths', 'quarters_eighths', 'sixteenths', 'halves_quarters_eighths_sixteenths', 'quarters_eighths_sixteenths_dots', 'sixteenths')

cp_choices = ('none', 'tight', 'dense', 'complex')

with open('sample_score.musicxml', 'w+') as f:
    key_number = 0
    rest_prob = np.random.choice([0, 0.2, 0.4, 0.6, 0.8], p=[0.5, 0.15, 0.15, 0.15, 0.05])
    treble_tp_key_choices = tight_tp_choices
    bass_tp_key_choices = wide_tp_choices
    treble_cp_key_choices = cp_choices
    bass_cp_key_choices = cp_choices
    f.write(str(generate_score(64, 16, 0, rest_prob, treble_tp_key_choices=treble_tp_key_choices, bass_tp_key_choices=bass_tp_key_choices, treble_cp_key_choices=treble_cp_key_choices, bass_cp_key_choices=bass_cp_key_choices)))