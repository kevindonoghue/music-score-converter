from data.generate_sample import generate_sample


generate_sample(2500, 5, 'quarters_chords_rests_c', 224, 224, 'quarters', 'quarters', 'complex', 'complex', 0.2, key_number_choices=(0,))
generate_sample(2500, 5, 'sixteenths_chords_rests_c', 224, 224, 'sixteenths', 'halves_wholes', 'none', 'complex', 0.2, key_number_choices=(0,))
generate_sample(2500, 5, 'halves_quarters_eighths_sixteenths_chords_rests_c', 224, 224, 'halves_quarters_eighths_sixteenths', 'quarters', 'none', 'dense', 0, key_number_choices=(0,))
generate_sample(2500, 5, 'halves_quarters_eighths_sixteenths_chords_rests_c', 224, 224, 'quarters', 'halves_quarters_eighths_sixteenths', 'dense', 'none', 0, key_number_choices=(0,))
